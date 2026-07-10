# PhD Integrator Project — TP2 D1

Pipeline de recherche reproductible pour l'etude d'un phenomene de transfert
de masse regi par une equation d'advection-diffusion-reaction singuliere :
calcul symbolique (SymPy), calcul numerique vectorise (NumPy/Polars),
acceleration HPC (Numba/Joblib), IA scientifique informee par la physique
(PINN, PyTorch) et visualisation de publication (Matplotlib/Plotly),
orchestres par Snakemake et valides en continu par GitHub Actions.

## 1. Procedure de deploiement (reproductibilite absolue, No-Sudo)

Le projet utilise [`uv`](https://github.com/astral-sh/uv), un gestionnaire de
paquets Python ecrit en Rust, garantissant un environnement **entierement
scelle dans l'espace utilisateur** (aucun droit root requis).

```bash
# 1. Installer uv (espace utilisateur uniquement, aucun sudo)
curl -LsSf https://astral.sh/uv/install.sh | sh        # Linux / macOS
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 2. Cloner le depot et se placer a la racine
git clone <url-du-depot>
cd phd_integrator_project

# 3. Synchroniser l'environnement virtuel isole a partir du verrouillage deterministe
uv sync --all-groups

# 4. Verifier la qualite du code (0 erreur exigee)
uv run ruff check src/ tests/
uv run mypy --strict src/

# 5. Executer la suite de tests avec couverture (>= 90% exigee)
uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=90 tests/

# 6. Executer le pipeline scientifique complet via Snakemake
uv run snakemake --cores 1 all
```

Le fichier `uv.lock` fige toutes les versions transitives : deux machines
executant `uv sync` obtiennent bit-a-bit le meme environnement, condition
necessaire a la reproductibilite scientifique sur une infrastructure tierce
(CI ou supercalculateur).

**Note d'environnement (Windows) :** `snakemake` ne declare officiellement
son support qu'a partir de Python 3.11 ; la dependance est donc marquee
`python_version >= '3.11'` dans `pyproject.toml`. Sous Python 3.10, le reste
du pipeline (Modules 3 a 8) reste entierement fonctionnel et teste — seule
l'orchestration Snakemake necessite 3.11+, ce qui est le cas des jobs CI
correspondants.

## 2. Architecture du depot

```
phd_integrator_project/
├── .github/workflows/ci_cd_pipeline.yml   # CI/CD : lint, types, tests, artefacts
├── data/raw_sensors/                      # Mesures climatiques synthetiques (Parquet)
├── src/
│   ├── symbolic_derivation.py             # Module 3 : SymPy -> residu -> lambdify
│   ├── numerical_core.py                  # Modules 4-5 : ndarray, ingestion, stabilite
│   ├── hpc_acceleration.py                # Module 6 : profiling, Numba JIT, Joblib
│   ├── deep_pinn.py                       # Module 7 : PINN PyTorch, autograd, device
│   ├── visualization.py                   # Module 8 : figures statiques/interactives
│   └── generate_artefacts.py              # Orchestration : entrainement + figures
├── tests/                                 # Suite PyTest (couverture >= 90%)
├── outputs/{figures,models}/              # Artefacts generes (PDF, HTML, .pt)
├── scripts/generate_synthetic_sensors.py  # Generateur du jeu de donnees synthetique
├── Snakefile                              # DAG : symbolic -> ingest -> stability -> pinn -> plots
├── pyproject.toml                         # Dependances (PEP 518) + config ruff/mypy/pytest
└── uv.lock                                # Verrouillage deterministe
```

## 3. Modele physique et resultats

Le systeme etudie est l'equation d'advection-diffusion :

```
du/dt + c * du/dx - nu * d2u/dx2 = f(x, t)
```

avec `c` le coefficient d'advection et `nu` le coefficient de diffusion. La
solution analytique candidate retenue est une **onde solitaire**
`u(x, t) = tanh(x - c*t)`. SymPy derive symboliquement les operateurs
differentiels exacts, ce qui permet de construire le terme source residuel
`f(x, t)` **exact** correspondant a cette solution — utilise ensuite comme
verite terrain pour entrainer et valider le PINN.

Le PINN (perceptron multicouche, activations `tanh`) apprend `u_hat(x, t)`
en minimisant une perte composite :

```
L = L_physique(residu de l'EDP via autograd) + L_donnees(ecart aux capteurs/CL)
```

Le residu physique est evalue **sans maillage** : les derivees
`du/dt`, `du/dx`, `d2u/dx2` sont obtenues par differentiation automatique
(`torch.autograd.grad`) directement aux points d'echantillonnage, plutot que
par difference finies discretes — ce qui evite toute erreur de troncature
liee a un schema numerique classique.

La matrice de Hilbert `H_ij = 1/(i+j-1)` sert de cas d'ecole pour illustrer
le mauvais conditionnement (`kappa(A)` croit exponentiellement avec `n`) :
une perturbation d'ordre `1e-7` sur le second membre peut etre amplifiee
jusqu'a un facteur `kappa(A)` sur la solution, conformement a la theorie de
perturbation lineaire `||dalpha||/||alpha|| <= kappa(A) * ||db||/||b||`.

## 4. Analyse critique : precision des flottants et acceleration

- **float16** : erreur de reconstruction du systeme lineaire non negligeable
  des `n >= 10` sur la matrice de Hilbert (mantisse 10 bits insuffisante
  face a `kappa(A)` qui explose) — impropre a tout calcul scientifique
  serieux au-dela de petites dimensions.
- **float32** : compromis memoire/precision acceptable pour l'inference PINN
  (poids du reseau), mais insuffisant pour l'algebre lineaire mal
  conditionnee sans repetition en double precision.
- **float64** : reference utilisee pour toute la partie symbolique/numerique
  du pipeline (`numerical_core.py`), necessaire pour que le residu
  `||Aα - b||` reste dans les tolerances `atol`/`rtol` de validation.
- **`fastmath=True` (Numba)** : autorise le compilateur LLVM a reordonner
  les operations flottantes (associativite, elimination de `NaN`/`Inf`
  checks), ce qui viole la stricte conformite IEEE 754 — gain de
  performance mesurable (voir `profile_filter`) au prix d'une reproductibilite
  bit-a-bit non garantie entre executions/architectures. A eviter des que la
  correction numerique prime sur la vitesse (ex. calcul du residu de
  validation).
- Les assertions de tests utilisent systematiquement `np.isclose` (jamais
  `==`) car l'arithmetique flottante IEEE 754 accumule des erreurs
  d'arrondi non nulles a chaque operation.

## 5. Validation experimentale du pipeline CI/CD (Module 9.2)

Procedure suivie pour demontrer que le pipeline detecte reellement les
regressions :

1. Une erreur de typage flagrante est introduite temporairement (ex. appel
   d'une fonction `numerical_core` attendant un `np.ndarray` avec une
   chaine `str`).
2. Le commit est pousse sur une branche ; le job `mypy --strict` du
   workflow `ci_cd_pipeline.yml` echoue immediatement, bloquant le badge de
   statut.
3. L'erreur est corrigee, un test de non-regression cible est ajoute dans
   `tests/`, et le correctif est documente via un commit respectant la
   norme **Conventional Commits** (ex. `fix(numerical-core): reject
   non-ndarray input in ingest_sensor_coordinates`).

## 6. Badges de statut et artefacts

```markdown
![CI/CD](https://github.com/<owner>/<repo>/actions/workflows/ci_cd_pipeline.yml/badge.svg)
```

Les artefacts scientifiques (figures PDF vectorielles, surface 3D HTML
interactive, poids du modele PINN `.pt`) sont publies automatiquement par le
job GitHub Actions via `actions/upload-artifact` et telechargeables depuis
l'onglet *Actions* de chaque execution reussie du pipeline.
