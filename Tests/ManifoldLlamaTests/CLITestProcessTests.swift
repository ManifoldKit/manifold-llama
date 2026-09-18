import Darwin
import Foundation
import XCTest

final class CLITestProcessTests: XCTestCase {
  private let shell = URL(fileURLWithPath: "/bin/sh")

  func test_outputBeyondPipeCapacityPreservesBothStreamsAndExitStatus() throws {
    let result = try CLITestProcess.run(
      executable: shell,
      arguments: ["-c", "dd if=/dev/zero bs=1048576 count=2 2>/dev/null; (dd if=/dev/zero bs=1048576 count=2 2>/dev/null) >&2; exit 7"],
      timeout: 10)
    XCTAssertEqual(result.exitCode, 7)
    XCTAssertEqual(result.stdout.utf8.count, 2 * 1024 * 1024)
    XCTAssertEqual(result.stderr.utf8.count, 2 * 1024 * 1024)
  }

  func test_timeoutIsReportedAndDirectChildIsGone() throws {
    let pidFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { remove(pidFile) }
    XCTAssertThrowsError(try CLITestProcess.run(
      executable: shell, arguments: ["-c", "echo $$ > '\(pidFile.path)'; exec sleep 30"], timeout: 0.3
    )) { XCTAssertEqual(String(describing: $0), CLITestProcess.Failure.timeout.description) }
    let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
    XCTAssertEqual(kill(pid, 0), -1)
    XCTAssertEqual(errno, ESRCH)
  }

  func test_cancellationIsReported() throws {
    var polls = 0
    XCTAssertThrowsError(try CLITestProcess.run(
      executable: shell, arguments: ["-c", "exec sleep 30"],
      cancelled: { polls += 1; return polls > 3 }
    )) { XCTAssertEqual(String(describing: $0), CLITestProcess.Failure.cancelled.description) }
  }

  func test_cleanupInspectionFailureStillKillsFrozenChildAndReportsOriginalTimeout() throws {
    let pidFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { remove(pidFile) }
    XCTAssertThrowsError(try CLITestProcess.run(
      executable: shell, arguments: ["-c", "echo $$ > '\(pidFile.path)'; exec sleep 30"], timeout: 0.3,
      beforeCleanupDiscovery: { throw CLITestProcess.Failure.inspection(-99) }
    )) {
      XCTAssertTrue(String(describing: $0).contains("execution deadline"))
      XCTAssertTrue(String(describing: $0).contains("-99"))
    }
    let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
    XCTAssertEqual(kill(pid, 0), -1, "cleanup failure must not leave SIGSTOP-only processes")
    XCTAssertEqual(errno, ESRCH)
  }

  func test_taskCancellationUsesDefaultCancellationHook() async throws {
    let pidFile = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { remove(pidFile) }
    let task = Task.detached {
      try CLITestProcess.run(executable: URL(fileURLWithPath: "/bin/sh"),
        arguments: ["-c", "echo $$ > '\(pidFile.path)'; exec sleep 30"])
    }
    let deadline = ContinuousClock.now.advanced(by: .seconds(5))
    while !FileManager.default.fileExists(atPath: pidFile.path), ContinuousClock.now < deadline {
      try await Task.sleep(for: .milliseconds(10))
    }
    task.cancel()
    do { _ = try await task.value; XCTFail("cancelled capture returned success") }
    catch { XCTAssertEqual(String(describing: error), CLITestProcess.Failure.cancelled.description) }
  }

  func test_outputLimitIsReported() throws {
    XCTAssertThrowsError(try CLITestProcess.run(
      executable: shell, arguments: ["-c", "dd if=/dev/zero bs=4096 count=2 2>/dev/null"], outputLimit: 1024
    )) { XCTAssertEqual(String(describing: $0), CLITestProcess.Failure.outputLimit.description) }
  }

  func test_captureSetupFailureIsNotSuccess() throws {
    let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    XCTAssertThrowsError(try CLITestProcess.run(executable: shell, arguments: ["-c", "exit 0"], captureRoot: missing))
  }

  func test_launchFailureIsNotSuccess() throws {
    XCTAssertThrowsError(try CLITestProcess.run(executable: URL(fileURLWithPath: "/missing-cli-\(UUID().uuidString)"), arguments: []))
  }

  // A native fixture calls setsid in the grandchild: killing the CLI's process
  // group alone cannot satisfy this cleanup assertion.
  func test_timeoutTerminatesDescendantInDifferentProcessGroup() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { remove(directory) }
    let source = directory.appendingPathComponent("fixture.c")
    let binary = directory.appendingPathComponent("fixture")
    let pidFile = directory.appendingPathComponent("child-pid")
    let code = """
      #include <unistd.h>
      #include <stdio.h>
      #include <signal.h>
      #include <sys/wait.h>
      int main(int argc, char **argv) {
        pid_t child = fork();
        if (child == 0) {
          pid_t grandchild = fork();
          if (grandchild < 0) return 4;
          if (grandchild > 0) { waitpid(grandchild, 0, 0); return 0; }
          if (setsid() < 0) return 2;
          FILE *f = fopen(argv[1], "w");
          if (!f) return 3;
          fprintf(f, "%d\\n", getpid()); fclose(f);
          signal(SIGTERM, SIG_IGN);
          for (;;) pause();
        }
        waitpid(child, 0, 0);
        return 0;
      }
      """
    try code.write(to: source, atomically: true, encoding: .utf8)
    let build = try CLITestProcess.run(executable: URL(fileURLWithPath: "/usr/bin/xcrun"), arguments: ["clang", source.path, "-o", binary.path], timeout: 30)
    XCTAssertEqual(build.exitCode, 0, build.stderr)
    XCTAssertThrowsError(try CLITestProcess.run(executable: binary, arguments: [pidFile.path], timeout: 0.5)) {
      XCTAssertEqual(String(describing: $0), CLITestProcess.Failure.timeout.description)
    }
    let pid = try XCTUnwrap(Int32(String(contentsOf: pidFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)))
    // Orphaned descendants may briefly be zombies until launchd reaps them;
    // neither a zombie nor ESRCH owns a running model or an open output handle.
    var info = proc_bsdinfo()
    let bytes = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, Int32(MemoryLayout<proc_bsdinfo>.size))
    XCTAssertTrue(bytes == 0 || info.pbi_status == UInt32(SZOMB), "descendant still running: \(pid)")
  }

  private func remove(_ url: URL) {
    do { try FileManager.default.removeItem(at: url) }
    catch { XCTFail("fixture cleanup failed: \(error)") }
  }
}
