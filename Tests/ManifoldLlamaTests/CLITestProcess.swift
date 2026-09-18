import Darwin
import Foundation

/// File-backed capture avoids the wait-before-drain deadlock with verbose model loaders.
/// Output size is polled: disk use can briefly overshoot the limit, but oversized
/// output is rejected before reading it into memory.
/// The owned CLI must keep children attached until they have been observed; daemonizing
/// (fork + immediate reparent) is intentionally outside this test helper's contract.
enum CLITestProcess {
  struct Result {
    let exitCode: Int32
    let stdout: String
    let stderr: String
  }

  enum Failure: Error, CustomStringConvertible {
    case timeout, cancelled, outputLimit, inspection(Int32), cleanup([Int32]), cleanupAfterFailure(String, String)
    var description: String {
      switch self {
      case .cleanupAfterFailure(let original, let cleanup): return "\(original); cleanup also failed: \(cleanup)"
      case .timeout: return "CLI exceeded its execution deadline"
      case .cancelled: return "CLI capture was cancelled"
      case .outputLimit: return "CLI exceeded its per-stream output limit"
      case .inspection(let pid): return "Could not inspect owned CLI process \(pid)"
      case .cleanup(let pids): return "CLI cleanup deadline expired for processes \(pids)"
      }
    }
  }

  private struct Identity: Hashable {
    let pid: pid_t
    let seconds: UInt64
    let microseconds: UInt64
    static func read(_ pid: pid_t) throws -> Identity? {
      var info = proc_bsdinfo()
      let size = Int32(MemoryLayout<proc_bsdinfo>.size)
      guard proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, size) == size else {
        if errno == ESRCH { return nil }
        throw Failure.inspection(pid)
      }
      if info.pbi_status == UInt32(SZOMB) { return nil }
      return Identity(pid: pid, seconds: info.pbi_start_tvsec, microseconds: info.pbi_start_tvusec)
    }
    func isAlive() throws -> Bool { try Self.read(pid) == self }
  }

  static func run(
    executable: URL, arguments: [String], timeout: TimeInterval = 180,
    outputLimit: UInt64 = 64 * 1024 * 1024,
    cancelled: () -> Bool = { Task<Never, Never>.isCancelled },
    captureRoot: URL = FileManager.default.temporaryDirectory,
    beforeCleanupDiscovery: () throws -> Void = {}
  ) throws -> Result {
    let fm = FileManager.default
    let directory = captureRoot.appendingPathComponent("llama-cli-\(UUID().uuidString)")
    try fm.createDirectory(at: directory, withIntermediateDirectories: false)
    defer {
      do { try fm.removeItem(at: directory) }
      catch { NSLog("CLI capture directory cleanup failed: %@", String(describing: error)) }
    }
    let stdoutURL = directory.appendingPathComponent("stdout")
    let stderrURL = directory.appendingPathComponent("stderr")
    try Data().write(to: stdoutURL)
    try Data().write(to: stderrURL)
    let stdout = try FileHandle(forWritingTo: stdoutURL)
    let stderr = try FileHandle(forWritingTo: stderrURL)
    defer {
      do { try stdout.close(); try stderr.close() }
      catch { NSLog("CLI capture handle cleanup failed: %@", String(describing: error)) }
    }
    let process = Process()
    process.executableURL = executable
    process.arguments = arguments
    process.currentDirectoryURL = directory
    process.standardOutput = stdout
    process.standardError = stderr
    try process.run()
    var owned = Set<Identity>()
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .seconds(timeout))
    do {
      if let root = try Identity.read(process.processIdentifier) { owned.insert(root) }
      while process.isRunning {
        try discover(&owned)
        if cancelled() { throw Failure.cancelled }
        if clock.now >= deadline { throw Failure.timeout }
        try checkSize(stdoutURL, stderrURL, limit: outputLimit)
        Thread.sleep(forTimeInterval: 0.02)
      }
      process.waitUntilExit()
      // A successful direct child must not leave a background inference process behind.
      try terminate(&owned, process: process, beforeDiscovery: beforeCleanupDiscovery)
      try checkSize(stdoutURL, stderrURL, limit: outputLimit)
      return Result(
        exitCode: process.terminationStatus,
        stdout: String(decoding: try Data(contentsOf: stdoutURL), as: UTF8.self),
        stderr: String(decoding: try Data(contentsOf: stderrURL), as: UTF8.self))
    } catch {
      let original = error
      do { try terminate(&owned, process: process, beforeDiscovery: beforeCleanupDiscovery) }
      catch { throw Failure.cleanupAfterFailure(String(describing: original), String(describing: error)) }
      throw original
    }
  }

  private static func checkSize(_ stdout: URL, _ stderr: URL, limit: UInt64) throws {
    for url in [stdout, stderr] {
      let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
      guard let size = attributes[.size] as? NSNumber else { throw Failure.outputLimit }
      if size.uint64Value > limit { throw Failure.outputLimit }
    }
  }

  private static func discover(_ owned: inout Set<Identity>, deadline: ContinuousClock.Instant? = nil) throws {
    var pending = Array(owned)
    while let parent = pending.popLast() {
      if let deadline, ContinuousClock.now >= deadline { throw Failure.cleanup(owned.map(\.pid)) }
      guard try parent.isAlive() else { continue }
      // proc_listchildpids returns a PID count (unlike proc_pidinfo); buffer size is bytes.
      let required = proc_listchildpids(parent.pid, nil, 0)
      guard required >= 0 else { throw Failure.inspection(parent.pid) }
      var pids = [pid_t](repeating: 0, count: max(64, Int(required) + 64))
      let count = pids.withUnsafeMutableBytes {
        proc_listchildpids(parent.pid, $0.baseAddress, Int32($0.count))
      }
      guard count >= 0, Int(count) < pids.count else {
        throw Failure.inspection(parent.pid)
      }
      for pid in pids.prefix(Int(count)) where pid > 0 {
        if let child = try Identity.read(pid), owned.insert(child).inserted { pending.append(child) }
      }
    }
  }

  private static func terminate(
    _ owned: inout Set<Identity>, process: Process, beforeDiscovery: () throws -> Void
  ) throws {
    let deadline = ContinuousClock.now.advanced(by: .seconds(3))
    // Freeze before the last recursive discovery so observed parents cannot fork
    // another child between the snapshot and termination. Identity checks prevent PID reuse.
    var frozen = Set<Identity>()
    var inspectionFailure: Error?
    do {
      repeat {
        for identity in owned.subtracting(frozen) where try identity.isAlive() {
          if kill(identity.pid, SIGSTOP) != 0 && errno != ESRCH { throw Failure.inspection(identity.pid) }
          frozen.insert(identity)
        }
        let before = owned.count
        try beforeDiscovery()
        try discover(&owned, deadline: deadline)
        if owned.count == before { break }
      } while true
    } catch { inspectionFailure = error }
    // An inspection failure must never strand processes that we already froze.
    for identity in owned {
      do {
        if try identity.isAlive(), kill(identity.pid, SIGKILL) != 0 && errno != ESRCH {
          inspectionFailure = Failure.inspection(identity.pid)
        }
      } catch {
        inspectionFailure = error
        // Frozen identities cannot exit/reuse their PID before our kill.
        if frozen.contains(identity) { kill(identity.pid, SIGKILL) }
      }
    }
    // If the initial inspection failed, Foundation still owns the direct child.
    if process.isRunning { kill(process.processIdentifier, SIGKILL) }
    while true {
      var remaining: [Identity] = []
      for identity in owned {
        do { if try identity.isAlive() { remaining.append(identity) } }
        catch { inspectionFailure = error; remaining.append(identity) }
      }
      if remaining.isEmpty && !process.isRunning {
        process.waitUntilExit()
        if let inspectionFailure { throw inspectionFailure }
        return
      }
      if ContinuousClock.now >= deadline { throw Failure.cleanup(remaining.map(\.pid)) }
      Thread.sleep(forTimeInterval: 0.01)
    }
  }
}
