import json
import unittest
from unittest import mock

from methods.p7 import run_m1_full as runner


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


class P7M1FullSafetyTests(unittest.TestCase):
    def test_v2_approval_matches_proposal_and_current_files(self):
        config = load(runner.CONFIG)
        proposal = load(runner.PROPOSAL)
        approved = load(runner.APPROVAL)

        self.assertEqual(runner.approval_payload(proposal), runner.approval_payload(approved))
        self.assertEqual(runner.digest(runner.approval_payload(approved)), "8cea962ee3c6a221bb7307799d53dcf016779440b4dbebdeeea281e399917718")
        self.assertEqual(runner.issues(config, approved), [])
        self.assertIn("execution record is not approved", runner.issues(config, proposal))

    def test_run_requires_explicit_confirm_start(self):
        with mock.patch.object(runner, "run", return_value=0) as delegated:
            with self.assertRaisesRegex(ValueError, "--confirm-start"):
                runner.main(["run"])

            self.assertEqual(runner.main(["run", "--confirm-start", "--run-id", "delegated-full"]), 0)
            delegated.assert_called_once_with("delegated-full")


if __name__ == "__main__":
    unittest.main()
