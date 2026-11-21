"""Synthetic document generator.

An offline stand-in for the real corpora. It lets the whole pipeline run with
no internet connection (demos, tests, throughput measurements), and because it
knows what it generated, it provides ground truth for testing the filters:
each document's ``meta["kind"]`` says whether it was meant to be clean,
a duplicate, boilerplate, and so on.

Synthetic text is template-based. It is good enough to exercise every pipeline
stage, but it is NOT training data: a model trained on it would mostly learn
the templates. Every synthetic record is labelled as such in its provenance
(source "synthetic", licence "synthetic") so it cannot be mistaken for real
text.

Two roles mirror the two retrieving agents:
  * "factory"   - mostly clean in-domain documents, like a curated repository;
  * "generator" - a noisy mix, like the open web: off-topic pages, boilerplate,
                  fragments, mark-up, other languages and many duplicates.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Iterator

from .models import RawDocument

_SLOT_RE = re.compile(r"\{(\w+)(?::(\d+):(\d+))?\}")

SURNAMES = ["Anderson", "Baker", "Carter", "Dalton", "Ellis", "Foster", "Grant", "Hughes",
            "Ingram", "Jensen", "Keller", "Lawson", "Morgan", "Norris", "Owens", "Parker",
            "Quinn", "Reyes", "Sutton", "Turner", "Vance", "Walsh", "Young", "Zimmer"]
FIRST_NAMES = ["Alice", "Brian", "Carla", "David", "Elena", "Frank", "Grace", "Henry",
               "Irene", "James", "Karen", "Louis", "Maria", "Nathan", "Olivia", "Peter"]
CITIES = ["London", "Paris", "Berlin", "Madrid", "Rome", "Vienna", "Dublin", "Oslo",
          "Tokyo", "Sydney", "Toronto", "Chicago", "Boston", "Denver", "Seattle", "Houston"]
COUNTRIES = ["England", "France", "Germany", "Spain", "Italy", "Austria", "Ireland",
             "Norway", "Japan", "Australia", "Canada", "Brazil", "India", "Mexico"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "June", "July", "September", "October"]

POOLS: dict[str, list[str]] = {
    "surname": SURNAMES, "first": FIRST_NAMES, "city": CITIES, "country": COUNTRIES,
    "weekday": WEEKDAYS, "month": MONTHS,
    # medical
    "condition": ["hypertension", "type 2 diabetes", "chronic kidney disease", "asthma",
                  "heart failure", "rheumatoid arthritis", "pneumonia", "breast cancer",
                  "atrial fibrillation", "osteoporosis", "sepsis", "migraine", "hepatitis C",
                  "chronic obstructive pulmonary disease", "ischemic stroke"],
    "drug": ["amlodipine", "metformin", "atorvastatin", "lisinopril", "prednisone", "apixaban",
             "salbutamol", "methotrexate", "losartan", "insulin glargine", "azithromycin",
             "tamoxifen"],
    "outcome": ["mortality", "hospital readmission", "cardiovascular events",
                "disease progression", "renal function decline", "symptom severity"],
    "population": ["adults", "elderly patients", "children", "pregnant women", "outpatients",
                   "hospitalized patients"],
    "design": ["randomized controlled trial", "prospective cohort study",
               "retrospective analysis", "double-blind placebo-controlled trial",
               "multicenter observational study"],
    "measure": ["systolic blood pressure", "serum creatinine", "glycated hemoglobin",
                "heart rate", "pain score", "forced expiratory volume", "LDL cholesterol"],
    # legal
    "reporter": ["U.S.", "F.2d", "F.3d", "N.E.2d", "P.2d", "A.2d", "S.W.2d", "Cal.", "N.Y."],
    "claim": ["breach of contract", "negligence", "wrongful termination", "trespass", "fraud",
              "defamation", "unjust enrichment"],
    "court": ["District Court", "Superior Court", "Circuit Court", "Court of Common Pleas",
              "Court of Appeals"],
    "error": ["excluding testimony", "admitting hearsay evidence", "instructing the jury",
              "denying the motion to dismiss", "granting summary judgment"],
    "act": ["Evidence Act", "Civil Practice Act", "Commercial Code", "Labor Relations Act",
            "Securities Act"],
    "fact": ["failed to deliver the goods on the agreed date",
             "knew of the defect and did not disclose it",
             "terminated the lease without the required notice",
             "entered the property without permission",
             "made the statement with knowledge of its falsity"],
    "ruling": ["excluded", "admitted", "decided", "resolved"],
    # scientific
    "field": ["statistical physics", "machine learning", "fluid dynamics",
              "condensed matter physics", "computational biology", "signal processing",
              "numerical analysis", "quantum optics"],
    "method": ["variational method", "Monte Carlo algorithm", "spectral approach",
               "gradient-based estimator", "Bayesian framework", "perturbation scheme"],
    "task": ["parameter estimation", "phase classification", "noise reduction",
             "structure prediction", "sparse recovery", "anomaly detection"],
    "quantity": ["variance", "correlation function", "energy spectrum", "density",
                 "probability distribution", "coupling coefficient"],
    "structure": ["matrix", "network", "lattice", "tensor", "graph"],
    "metric": ["estimation error", "computation time", "variance", "energy consumption"],
    "journal": ["Physical Review", "Journal of Computational Physics", "Neural Computation",
                "Annals of Statistics", "Nature Communications"],
    # news
    "agency": ["Reuters", "AP", "AFP"],
    "org": ["the Central Bank", "the Health Ministry", "the Transport Authority",
            "the Energy Commission", "the Trade Department", "the National Statistics Office",
            "the City Council", "the Education Ministry"],
    "title": ["Governor", "Minister", "Chairman", "Director", "Commissioner", "Senator"],
    "event": ["interest rates would rise by a quarter point",
              "a new rail line would open next year",
              "fuel subsidies would be phased out",
              "emergency funding had been approved",
              "the merger had been cleared by regulators",
              "exports had reached a record level"],
    "stat": ["wages", "consumer prices", "industrial output", "unemployment claims",
             "housing starts", "retail sales"],
    "bank": ["Barclays", "Nomura", "Citigroup", "Deutsche Bank", "Macquarie"],
    "quote": ["We will act again if the pressure persists through the winter",
              "This decision reflects months of careful work",
              "The public deserves a clear explanation",
              "We expect the effects to be visible within the year",
              "No one should doubt our commitment to stability"],
    # general
    "prefix": ["North", "East", "West", "South", "Green", "Stone", "River", "Oak", "Mill",
               "Spring", "Clear", "High"],
    "suffix": ["field", "brook", "ford", "ton", "haven", "bridge", "wood", "dale", "port"],
    "place_type": ["town", "city", "municipality", "village", "district"],
    "river": ["Avon", "Tamar", "Severn", "Clyde", "Shannon", "Elbe", "Loire"],
    "industry": ["agriculture", "textile manufacturing", "tourism", "shipbuilding", "mining",
                 "fishing", "wine production", "steel production"],
    "building": ["cathedral", "university", "library", "railway station", "town hall",
                 "observatory"],
    # code
    "verb": ["filter", "count", "merge", "normalize", "collect", "scale", "group", "trim"],
    "noun": ["records", "values", "tokens", "entries", "samples", "rows", "items", "scores"],
    "arg": ["limit", "factor", "threshold", "offset", "width", "step"],
}


def fill(template: str, rng: random.Random) -> str:
    """Replace {slot} with a random choice and {int:a:b} / {dec:a:b} / {year} with numbers."""

    def replace(match: re.Match[str]) -> str:
        name, low, high = match.group(1), match.group(2), match.group(3)
        if name == "int":
            return str(rng.randint(int(low), int(high)))
        if name == "dec":
            return f"{rng.uniform(int(low), int(high)):.1f}"
        if name == "year":
            return str(rng.randint(1975, 2023))
        if name.endswith("_c"):                       # capitalised variant
            value = rng.choice(POOLS[name[:-2]])
            return value[0].upper() + value[1:]
        return rng.choice(POOLS[name])

    return _SLOT_RE.sub(replace, template)


def pick(rng: random.Random, *options: str) -> str:
    return fill(rng.choice(options), rng)


# ---------------------------------------------------------------------------
# In-domain documents
# ---------------------------------------------------------------------------

def medical(rng: random.Random) -> str:
    parts = [
        "BACKGROUND: " + pick(
            rng,
            "{condition_c} is a major cause of {outcome} in {population}.",
            "The optimal treatment of {condition} in {population} remains uncertain.",
            "Few studies have evaluated the effect of {drug} on {outcome} in patients with {condition}.",
        ),
        "OBJECTIVE: " + pick(
            rng,
            "The aim of this study was to evaluate the efficacy and safety of {drug} in patients with {condition}.",
            "We investigated whether {drug} reduces {outcome} in {population} with {condition}.",
        ),
        "METHODS: " + pick(
            rng,
            "We conducted a {design} in {int:80:900} patients with {condition}. Patients were randomized to receive {drug} {int:5:80} mg daily or placebo for {int:8:52} weeks.",
            "A total of {int:120:2400} patients with {condition} were enrolled at {int:3:20} hospitals between {year} and {year}. {measure_c} was measured at baseline and at each follow-up visit.",
            "This {design} included {int:60:1500} {population} diagnosed with {condition}. The primary outcome was {outcome} at {int:6:36} months.",
        ),
        "RESULTS: " + pick(
            rng,
            "Mean {measure} decreased by {dec:2:20} in the treatment group compared with {dec:0:5} in the placebo group (p < 0.0{int:1:5}). Adverse effects were observed in {int:2:15}% of treated patients.",
            "{outcome_c} occurred in {int:5:30}% of patients receiving {drug} and {int:31:45}% of controls (hazard ratio 0.{int:45:85}; 95% CI 0.{int:30:44} to 0.{int:86:99}).",
            "After {int:6:36} months of follow-up, {measure} was significantly lower in the {drug} group (p = 0.0{int:1:4}), and the incidence of {outcome} was reduced by {int:10:40}%.",
        ),
        "CONCLUSIONS: " + pick(
            rng,
            "These results suggest that {drug} significantly reduced {outcome} in patients with {condition}, with an acceptable rate of adverse outcomes.",
            "In conclusion, treatment with {drug} was associated with improved {measure} and a lower risk of {outcome}.",
            "These findings indicate that {drug} is an effective therapy for {condition} in {population}; larger trials are needed to confirm the survival benefit.",
        ),
    ]
    return " ".join(parts)


def legal(rng: random.Random) -> str:
    body = [
        "Appeal from the {court}. The plaintiff brought this action against the defendant for {claim} and sought damages of ${int:5:900},000.",
        "The trial court entered judgment for the defendant, and the plaintiff appealed.",
        "The appellant contends that the court erred in {error} under Section {int:2:400} of the {act}.",
        "We have reviewed the record and the authorities cited, including {surname} v. {surname}, {int:100:600} {reporter} {int:1:999}.",
        "The statute provides no remedy in equity where an adequate remedy at law exists. See {surname} v. {surname}, {int:100:600} {reporter} {int:1:999}, {int:1:999}.",
        "The evidence at trial showed that the defendant {fact}. The jury returned a verdict for the plaintiff in the amount of ${int:2:400},000.",
        "The respondent argues that the claim is barred by the statute of limitations, citing {surname} v. {surname}, {int:100:600} {reporter} {int:1:999} ({year}).",
        "We hold that the question was properly {ruling} and that the verdict is supported by the evidence.",
        "Counsel for the appellee conceded at oral argument that the contract contained no such provision.",
    ]
    rng.shuffle(body)
    sentences = [fill("{surname} v. {surname}, {int:100:600} {reporter} {int:1:999} ({year}).", rng)]
    sentences += [fill(s, rng) for s in body[: rng.randint(6, 8)]]
    sentences.append(pick(
        rng,
        "The judgment of the {court} is affirmed.",
        "Reversed and remanded for further proceedings consistent with this opinion.",
        "The order is vacated and the petition is dismissed. It is so ordered.",
    ))
    sentences.append(fill("Opinion of the court delivered by {surname}, J.", rng))
    return " ".join(sentences)


def scientific(rng: random.Random) -> str:
    parts = [
        "Abstract. We propose a new {method} for {task} in {field}, and we show that it remains stable in the presence of noise.",
        "Introduction. {task_c} is a central problem in {field} [1, 2]. Previous approaches rely on a {method} ({surname} et al., {year}) or on direct numerical simulation [3], which becomes expensive for large systems.",
        "In this paper we derive an estimator based on the {quantity} of the observed data, and we present an analysis of its convergence.",
        "Methods. The model is defined by a {structure} of coupling parameters, and the solution is obtained by optimization of the energy function, as shown in Fig. {int:1:4}. The equations are solved with a {method} described by {surname} et al. [4].",
        "Results. Experiments on {int:3:9} benchmark datasets demonstrate that the proposed algorithm reduces the {metric} by {int:8:45}% compared with the method of {surname} et al. [5]. The measured {quantity} agrees with the theoretical distribution within statistical error.",
        "The convergence analysis is given in Section {int:2:5}, and the results are summarised in Table {int:1:3}.",
        "Discussion. The observed scaling of the {quantity} is consistent with the linear approximation, although the {metric} increases for systems with more than {int:100:900} parameters [6].",
        "Conclusion. The framework extends to higher dimensional systems and to other problems in {field}.",
        "References. [1] {first} {surname}, {journal} {int:10:99}, {int:100:999} ({year}). [2] {first} {surname} and {first} {surname}, {journal} {int:10:99}, {int:100:999} ({year}). [3] {first} {surname} et al., {journal} {int:10:99}, {int:100:999} ({year}).",
    ]
    return " ".join(fill(p, rng) for p in parts)


_CODE_TEMPLATES = [
    '''def {verb}_{noun}({noun}, {arg}):
    """Return the {noun} that satisfy the {arg} condition."""
    result = []
    for item in {noun}:
        # Keep only entries above the configured {arg}.
        if item > {arg}:
            result.append(item * {int:2:9})
    return result
''',
    '''def {verb}_{noun}({noun}, {arg}={int:1:20}):
    """Compute a running total of {noun} in windows of size {arg}."""
    total = 0
    output = []
    for index, value in enumerate({noun}):
        total += value
        if index >= {arg}:
            total -= {noun}[index - {arg}]
        # Emit one value per completed window.
        if index >= {arg} - 1:
            output.append(total / {arg})
    return output
''',
    '''class {verb_c}{noun_c}:
    """Tracks {noun} and reports a summary on request."""

    def __init__(self, {arg}):
        self.{arg} = {arg}
        self.{noun} = []

    def add(self, value):
        # Ignore values outside the allowed {arg}.
        if value <= self.{arg}:
            self.{noun}.append(value)

    def summary(self):
        if not self.{noun}:
            return {int:0:1}
        return sum(self.{noun}) / len(self.{noun})
''',
    '''/**
 * {verb_c} the {noun} using the given {arg}.
 */
function {verb}{noun_c}({noun}, {arg}) {
    const output = [];
    for (let i = 0; i < {noun}.length; i++) {
        // Skip entries that fall below the {arg}.
        if ({noun}[i] >= {arg}) {
            output.push({noun}[i] + {int:1:50});
        }
    }
    return output;
}
''',
    '''/**
 * Returns the number of {noun} greater than the {arg}.
 */
public static int {verb}{noun_c}(int[] {noun}, int {arg}) {
    int count = 0;
    for (int value : {noun}) {
        // Only values above the {arg} are counted.
        if (value > {arg}) {
            count += {int:1:5};
        }
    }
    return count;
}
''',
]


def code(rng: random.Random) -> str:
    functions = [fill(t, rng) for t in rng.sample(_CODE_TEMPLATES[:3], rng.randint(1, 2))]
    if rng.random() < 0.4:
        functions = [fill(rng.choice(_CODE_TEMPLATES[3:]), rng)]
    return "\n\n".join(functions)


def news(rng: random.Random) -> str:
    body = [
        "{title} {first} {surname} told reporters that the measure had the full support of the cabinet.",
        "\"{quote},\" {surname} said at a press conference in {city}.",
        "According to the National Statistics Office, {stat} rose {dec:1:6} percent over the same period.",
        "A spokesman for {org} said the change would take effect in {month}.",
        "Analysts at {bank} said the move was widely expected.",
        "The decision follows {int:3:18} months of negotiations, officials said on {weekday}.",
        "Opposition leader {first} {surname} said the plan would cost taxpayers ${int:2:90} million.",
        "\"{quote},\" the {title} added in a statement released later in the day.",
        "Markets in {city} and {city} closed slightly lower on {weekday}, traders reported.",
    ]
    rng.shuffle(body)
    lead = "{city_u}, {country} ({agency}) -- {org_c} announced on {weekday} that {event}, officials said."
    lead = lead.replace("{city_u}", rng.choice(CITIES).upper())
    org = rng.choice(POOLS["org"])
    lead = lead.replace("{org_c}", org[0].upper() + org[1:])
    return " ".join([fill(lead, rng)] + [fill(s, rng) for s in body[: rng.randint(6, 8)]])


def general(rng: random.Random) -> str:
    name = rng.choice(POOLS["prefix"]) + rng.choice(POOLS["suffix"])
    parts = [
        f"{name} is a {{place_type}} in {{country}}, located on the River {{river}} about {{int:8:140}} kilometres from {{city}}.",
        "It was founded in {int:1100:1890} and had a population of {int:3:480},{int:100:999} at the {year} census.",
        f"The economy of {name} was historically based on {{industry}} and {{industry}}, and the area is known as a centre for {{industry}}.",
        "The {building} was established in {int:1500:1950} by {first} {surname}, and it remains one of the oldest in the region.",
        f"According to local records, {name} received its charter in {{int:1200:1800}}, which is the reason the annual market is still held in {{month}}.",
        "The surrounding district consists of farmland and forest, with an average elevation of {int:40:900} metres.",
        f"Notable people born in {name} include the engineer {{first}} {{surname}} and the painter {{first}} {{surname}}.",
        "The {place_type} is served by a railway line that opened in {int:1840:1930}, and a regional airport lies {int:10:60} kilometres to the north.",
    ]
    first, rest = parts[0], parts[1:]
    rng.shuffle(rest)
    return " ".join(fill(p, rng) for p in [first] + rest[: rng.randint(5, 7)])


DOMAIN_BUILDERS: dict[str, Callable[[random.Random], str]] = {
    "general": general,
    "news": news,
    "medical": medical,
    "legal": legal,
    "scientific": scientific,
    "code": code,
}


# ---------------------------------------------------------------------------
# Noise, as found on the open web
# ---------------------------------------------------------------------------

def boilerplate(rng: random.Random) -> str:
    lines = [
        "Welcome to our website. Click here to read more about our products.",
        "Subscribe to our newsletter and enter your email to receive special offers.",
        "We use cookies to improve your experience. Accept all cookies or read our cookie policy.",
        "Free shipping on every order. Add to cart today and create an account to save.",
        "Read our privacy policy and terms of service. All rights reserved.",
        "Follow us on social media and share this article with your friends.",
        "You must be logged in to leave a comment. Forgot your password? Click here.",
    ]
    rng.shuffle(lines)
    return " ".join(lines[: rng.randint(5, 7)])


def fragment(rng: random.Random) -> str:
    return pick(rng, "Posted by {first} on {weekday}.", "Page {int:1:90} of {int:91:200}.",
                "Thanks for the update!", "{city} weather: {int:1:30} degrees.")


def markup(rng: random.Random) -> str:
    cell = '<div class="row"><p>&nbsp;</p><span id="c{int:1:999}">{noun}</span><br/></div>'
    return " ".join(fill(cell, rng) for _ in range(rng.randint(15, 30)))


_FOREIGN = [
    "El ayuntamiento anunció el lunes que las obras de la nueva línea de metro comenzarán el próximo año, según informaron fuentes municipales. Los vecinos de la zona han expresado su preocupación por el ruido y por los cortes de tráfico previstos durante los trabajos, que se prolongarán al menos dieciocho meses. El concejal de transportes aseguró que el proyecto cuenta con financiación suficiente y que se respetarán los plazos establecidos en el contrato.",
    "Le conseil municipal a annoncé lundi que les travaux de la nouvelle ligne de tramway débuteront l'année prochaine, selon des sources proches du dossier. Les habitants du quartier ont exprimé leur inquiétude face au bruit et aux perturbations de la circulation prévues pendant le chantier, qui devrait durer au moins dix-huit mois. L'adjoint aux transports a assuré que le projet dispose d'un financement suffisant et que les délais seront respectés.",
    "Der Stadtrat hat am Montag bekannt gegeben, dass die Arbeiten an der neuen Straßenbahnlinie im nächsten Jahr beginnen werden. Die Anwohner haben ihre Besorgnis über den Lärm und die Verkehrsbehinderungen geäußert, die während der Bauzeit von mindestens achtzehn Monaten zu erwarten sind. Der Verkehrsdezernent versicherte, dass das Projekt ausreichend finanziert ist und die vertraglich vereinbarten Fristen eingehalten werden.",
]


def foreign(rng: random.Random) -> str:
    return rng.choice(_FOREIGN)


def repetitive(rng: random.Random) -> str:
    return " ".join([fill("Buy cheap {noun} online today and save money on every order.", rng)] * 25)


NOISE_BUILDERS = {
    "boilerplate": boilerplate,
    "fragment": fragment,
    "markup": markup,
    "foreign_language": foreign,
    "repetitive": repetitive,
}

# kind -> share of documents, per role
MIX = {
    "factory": {"clean": 0.82, "near_duplicate": 0.08, "exact_duplicate": 0.04,
                "off_topic": 0.03, "fragment": 0.03},
    "generator": {"clean": 0.30, "off_topic": 0.25, "boilerplate": 0.10, "fragment": 0.07,
                  "markup": 0.05, "foreign_language": 0.05, "repetitive": 0.05,
                  "near_duplicate": 0.08, "exact_duplicate": 0.05},
}


class SyntheticGenerator:
    """Endless, reproducible stream of synthetic documents for one domain."""

    def __init__(self, domain: str, role: str = "factory", seed: int = 0) -> None:
        if domain not in DOMAIN_BUILDERS:
            raise ValueError(f"unknown domain '{domain}'")
        if role not in MIX:
            raise ValueError(f"unknown role '{role}' (use 'factory' or 'generator')")
        self.domain = domain
        self.role = role
        self.seed = seed
        mix = MIX[role]
        self._kinds = list(mix)
        self._weights = list(mix.values())

    def _edit_slightly(self, text: str, rng: random.Random) -> str:
        """A near-duplicate: the same document with one small change.

        The change is kept small enough that the copy really is a near
        duplicate at the default threshold (word 5-gram similarity of 0.85).
        Replacing one word alters five 5-grams, which is too much for a short
        document, so short documents get one word added at the end instead.
        """
        word = rng.choice(["updated", "revised", "notably", "overall"])
        words = text.split(" ")
        if len(words) < 150:
            return text.rstrip() + ("\n# " if self.domain == "code" else " ") + word
        words[rng.randrange(len(words))] = word
        return " ".join(words)

    def documents(self, skip: int = 0) -> Iterator[RawDocument]:
        rng = random.Random(self.seed)
        others = [d for d in DOMAIN_BUILDERS if d != self.domain]
        recent: list[str] = []          # clean documents that duplicates are copied from
        index = 0
        while True:
            kind = rng.choices(self._kinds, self._weights)[0]
            if kind in ("near_duplicate", "exact_duplicate") and not recent:
                kind = "clean"
            if kind == "clean":
                text = DOMAIN_BUILDERS[self.domain](rng)
                recent.append(text)
                if len(recent) > 200:
                    recent.pop(rng.randrange(len(recent)))
            elif kind == "near_duplicate":
                text = self._edit_slightly(rng.choice(recent), rng)
            elif kind == "exact_duplicate":
                text = rng.choice(recent)
            elif kind == "off_topic":
                text = DOMAIN_BUILDERS[rng.choice(others)](rng)
            else:
                text = NOISE_BUILDERS[kind](rng)
            if index >= skip:
                yield RawDocument(
                    text=text,
                    source="synthetic",
                    agent=self.role,
                    source_id=f"{self.domain}-{self.role}-{index}",
                    license="synthetic",
                    authority=0.5,
                    meta={"kind": kind},
                )
            index += 1
