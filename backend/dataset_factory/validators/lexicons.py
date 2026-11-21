"""Domain vocabularies used to measure terminology density.

These are deliberately compact lists of high-frequency domain words. They are
a cheap first signal; the learned classifier added in a later module makes the
finer judgement.
"""

from __future__ import annotations


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


MEDICAL = _words(
    """
    patient patients clinical clinically treatment treatments treated therapy
    therapies therapeutic disease diseases disorder disorders syndrome diagnosis
    diagnosed diagnostic symptoms symptom acute chronic dose doses dosage drug
    drugs medication medications surgery surgical operative postoperative
    preoperative tumor tumors tumour tumours cancer carcinoma malignant benign
    metastasis metastatic lesion lesions infection infections infected bacterial
    viral virus antibiotic antibiotics vaccine vaccination immune immunity
    antibody antibodies antigen inflammation inflammatory blood plasma serum
    cardiac heart cardiovascular coronary artery arterial vascular renal kidney
    hepatic liver pulmonary lung lungs respiratory gastric intestinal bowel
    neurological neural brain cerebral spinal nerve muscle muscular bone skeletal
    skin tissue tissues cell cells cellular membrane protein proteins enzyme
    enzymes receptor receptors hormone hormones insulin glucose diabetes diabetic
    hypertension pressure obesity pregnancy pregnant fetal maternal neonatal
    infant infants children pediatric adult adults elderly mortality morbidity
    survival prognosis incidence prevalence risk epidemiology hospital hospitals
    physician physicians nurse nursing care healthcare medical medicine health
    placebo randomized randomised trial trials cohort biopsy imaging mri ct
    radiography ultrasound pathology pathological histological anatomy
    physiological pharmacological toxicity adverse efficacy outcome outcomes
    pain fever injury injuries wound fracture bleeding hemorrhage stroke
    seizure anesthesia transplant transplantation dialysis chemotherapy
    radiotherapy mg kg ml mmhg vivo vitro gene genes genetic mutation mutations
    dna rna expression plasma urine urinary oral intravenous injection
    administered administration subjects healthy abnormal lymph lymphocytes
    platelet hemoglobin cholesterol lipid calcium sodium potassium rats mice
    """
)

LEGAL = _words(
    """
    court courts judge judges justice justices jury juror jurors trial appeal
    appeals appellate appellant appellants appellee appellees plaintiff
    plaintiffs defendant defendants petitioner petitioners respondent
    respondents counsel attorney attorneys statute statutes statutory
    constitution constitutional amendment jurisdiction judgment judgments
    verdict decree order ordered ruling ruled opinion dissent dissenting
    concurring affirmed reversed remanded vacated dismissed denied granted
    overruled sustained held holding motion motions complaint petition
    indictment conviction convicted sentence sentenced acquitted guilty
    evidence testimony witness witnesses deposition affidavit exhibit
    contract contracts agreement breach damages liability liable negligence
    tort fraud property deed lease mortgage creditor debtor bankruptcy
    injunction relief remedy equity claim claims suit lawsuit litigation
    proceeding proceedings hearing prosecution prosecutor defense objection
    precedent doctrine law laws legal lawful unlawful legislature legislative
    act section sections clause provision provisions pursuant thereof herein
    hereinafter aforesaid therein whereas supra infra habeas corpus certiorari
    writ mandamus estoppel fiduciary trustee executor heirs estate probate
    felony misdemeanor criminal civil rights party parties error assignment
    """
)

SCIENTIFIC = _words(
    """
    analysis analyses analyzed data dataset datasets model models modeling
    method methods methodology approach results result experiment experiments
    experimental theory theoretical hypothesis measured measurement
    measurements observed observation observations parameter parameters
    function functions equation equations distribution distributions sample
    samples sampling significant significance algorithm algorithms energy
    simulation simulations numerical compute computed computation estimate
    estimated estimation variance correlation regression probability
    statistical statistics structure structures system systems process
    processes properties property effect effects phase temperature density
    spectrum spectra frequency field fields particle particles quantum
    molecular molecule molecules atomic electron electrons optical magnetic
    mass velocity coefficient coefficients matrix vector vectors linear
    nonlinear dimensional dimension scale scaling order approximation
    solution solutions boundary condition conditions constant constants
    proposed propose present demonstrate investigate consider derived obtain
    obtained defined corresponding respectively furthermore therefore thus
    whereas framework performance evaluation benchmark training network
    networks neural learning optimization convergence theorem lemma proof
    corollary figure fig table section literature previous previously study
    studies research
    """
)

CODE_KEYWORDS = _words(
    """
    def return if else elif for while import from class self none true false
    try except finally raise with as lambda yield pass break continue in is
    not and or function var let const new this null undefined typeof
    public private protected static void int string boolean double float long
    char final abstract interface extends implements throws throw catch package
    func struct type map range defer go chan select switch case default
    fn mut impl pub use mod match enum trait echo array foreach endif
    """
)
