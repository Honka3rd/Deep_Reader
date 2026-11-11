"""Exercise the review entrypoint in disposable real Git repositories."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("commit_review.sh")


class CommitReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="review commit ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        self.git("init", "-b", "task")
        self.git("config", "user.name", "Review test")
        self.git("config", "user.email", "review@example.invalid")
        (self.root / "scripts").mkdir()
        shutil.copy2(SCRIPT, self.root / "scripts/commit_review.sh")
        (self.root / "tracked.txt").write_text("base\n")
        self.git("add", "scripts/commit_review.sh", "tracked.txt")
        self.git("commit", "-m", "base")
        self.before = self.git("rev-parse", "HEAD").stdout
        self.message = self.root / "message file.txt"
        self.message.write_text("Review change\n\nLiteral `code` and $(not-a-command).\n")

    def git(self, *args, check=True):
        return subprocess.run(["git", *args], cwd=self.root, env=self.env,
                              text=True, capture_output=True, check=check)

    def run_script(self, *args, check=True):
        return subprocess.run(
            ["bash", str(self.root / "scripts/commit_review.sh"),
             "--message-file", "../message file.txt", *args],
            cwd=self.root / "scripts", env=self.env,
            text=True, capture_output=True, check=check,
        )

    def stage(self):
        (self.root / "tracked.txt").write_text("staged\n")
        self.git("add", "tracked.txt")

    def test_commits_only_index_preserving_unstaged_and_untracked_files(self):
        self.stage()
        (self.root / "tracked.txt").write_text("unstaged\n")
        (self.root / "untracked.txt").write_text("private\n")
        result = self.run_script()
        self.assertIn("Local review commit:", result.stdout)
        self.assertEqual(self.git("show", "HEAD:tracked.txt").stdout, "staged\n")
        self.assertEqual((self.root / "tracked.txt").read_text(), "unstaged\n")
        self.assertNotEqual(self.git("show", "HEAD:untracked.txt", check=False).returncode, 0)
        self.assertEqual(self.git("log", "-1", "--format=%B").stdout.strip(),
                         self.message.read_text().strip())
        self.assertEqual(self.git("rev-parse", "HEAD^").stdout, self.before)

    def test_preview_preserves_index_and_head_and_does_not_run_hooks(self):
        self.stage()
        hook = self.root / ".git/hooks/pre-commit"
        hook.write_text("#!/bin/sh\ntouch hook-ran\nexit 1\n")
        hook.chmod(0o755)
        index = self.git("write-tree").stdout
        self.run_script("--dry-run")
        self.assertEqual(self.git("rev-parse", "HEAD").stdout, self.before)
        self.assertEqual(self.git("write-tree").stdout, index)
        self.assertFalse((self.root / "hook-ran").exists())

    def test_hook_failure_is_not_bypassed_or_reported_as_commit(self):
        self.stage()
        hook = self.root / ".git/hooks/pre-commit"
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        result = self.run_script(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Local review commit:", result.stdout)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout, self.before)
        self.assertEqual(self.git("diff", "--cached", "--name-only").stdout, "tracked.txt\n")

    def test_empty_index_does_not_stage_worktree(self):
        (self.root / "tracked.txt").write_text("unstaged\n")
        result = self.run_script(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No staged changes", result.stderr)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout, self.before)
        self.assertEqual(self.git("diff", "--cached", "--name-only").stdout, "")

    def test_detached_head_is_rejected(self):
        self.git("checkout", "--detach")
        self.stage()
        result = self.run_script(check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Detached HEAD", result.stderr)

    def test_staged_whitespace_errors_are_rejected(self):
        (self.root / "tracked.txt").write_text("trailing space \n")
        self.git("add", "tracked.txt")
        self.assertNotEqual(self.run_script(check=False).returncode, 0)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout, self.before)

    def test_linked_worktree_commits_its_own_branch_only(self):
        linked = self.root / "linked checkout"
        self.git("worktree", "add", "-b", "review-worktree", str(linked))
        (linked / "tracked.txt").write_text("linked change\n")
        self.git("-C", str(linked), "add", "tracked.txt")
        subprocess.run(
            ["bash", str(linked / "scripts/commit_review.sh"),
             "--message-file", str(self.message)],
            cwd=self.root, env=self.env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(self.git("rev-parse", "HEAD").stdout, self.before)
        self.assertEqual(self.git("show", "review-worktree:tracked.txt").stdout,
                         "linked change\n")

    def test_merge_conflict_and_resolved_pending_merge_are_rejected(self):
        self.git("checkout", "-b", "other")
        (self.root / "tracked.txt").write_text("other\n")
        self.git("commit", "-am", "other change")
        self.git("checkout", "task")
        self.stage()
        self.git("commit", "-m", "task change")
        self.assertNotEqual(self.git("merge", "other", check=False).returncode, 0)
        for resolved in (False, True):
            with self.subTest(resolved=resolved):
                if resolved:
                    self.stage()
                result = self.run_script(check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unfinished Git operation", result.stderr)

    def test_invalid_arguments_and_message_are_rejected(self):
        self.stage()
        for args in (("--amend",), ("--message-file", "duplicate")):
            self.assertNotEqual(self.run_script(*args, check=False).returncode, 0)
        self.message.unlink()
        self.assertNotEqual(self.run_script(check=False).returncode, 0)
        self.assertEqual(self.git("rev-parse", "HEAD").stdout, self.before)


if __name__ == "__main__":
    unittest.main()
