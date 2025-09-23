#!/usr/bin/env python3

# Owner(s): ["module: cpp"]

from pathlib import Path

import torch
from torch.testing._internal.common_utils import (
    install_cpp_extension,
    IS_WINDOWS,
    run_tests,
    TestCase,
)


# TODO: Fix this error in Windows:
# LINK : error LNK2001: unresolved external symbol PyInit__C
if not IS_WINDOWS:

    class TestDummyType(TestCase):
        @classmethod
        def setUpClass(cls):
            try:
                import dummy_type_test  # noqa: F401
            except Exception:
                install_cpp_extension(extension_root=Path(__file__).parent)

        def test_placeholder(self):
            """Placeholder test for version-aware DummyType conversions"""
            import dummy_type_test

            out = dummy_type_test.ops.test_fn(torch.tensor([1, 2, 3]), torch._C.Dummy(42))
            print(out)


if __name__ == "__main__":
    run_tests()
