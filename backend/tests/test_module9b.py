"""Tests for Module 9b: data-quality changes (medical cues, shorter code)."""

from __future__ import annotations

from dataset_factory.config import load_config
from dataset_factory.filters import GlobalFilter
from dataset_factory.validators import build_validator

# An older, translated PubMed abstract: no section labels, no "we ..." phrases.
OLD_ABSTRACT = (
    "[Reactivity of the sympathetic-adrenal system in juvenile thyrotoxicosis]. In "
    "examination of 48 patients with thyrotoxicosis, aged from 14 to 24 years, there was "
    "revealed a reduction of the activity of the mediatory link of the sympathico-adrenal "
    "system without any connection with age, the severity and duration of the disease, and "
    "also with the efficacy of drug therapy. The functional test with insulin in patients "
    "with thyrotoxicosis showed a significant reduction of adrenaline excretion (by 31%) "
    "compared with the control group of 20 healthy persons. The level of cortisol in the "
    "blood was elevated in 12 patients. Treatment with thyrostatic drugs normalized the "
    "hormone concentration in most cases."
)

GO_FUNC = (
    "// Close releases the resources held by the reader.\n"
    "func (r *Reader) Close() error {\n"
    "\tif r.closed {\n"
    "\t\treturn nil\n"
    "\t}\n"
    "\tr.closed = true\n"
    "\treturn r.file.Close()\n"
    "}\n"
)


def test_medical_accepts_old_unlabelled_abstract():
    config = load_config()
    result = build_validator("medical", config).validate_text(OLD_ABSTRACT)
    assert result.passed, result.details


def test_medical_still_rejects_web_text():
    config = load_config()
    text = (
        "Our shop sells the best vitamins in town. Come and visit us today, we have "
        "great prices and friendly staff who will help you find what you need. " * 6
    )
    assert not build_validator("medical", config).validate_text(text).passed


def test_code_filter_keeps_short_functions():
    config = load_config()
    assert config.global_filter_for("code").min_chars < config.global_filter_for("general").min_chars
    assert len(GO_FUNC) < 200
    assert GlobalFilter(config.global_filter_for("code")).check_text(GO_FUNC).passed
