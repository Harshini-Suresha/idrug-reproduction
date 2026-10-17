# Put the real data here

Drop these 5 files in this folder (same names). Every file is a plain CSV with a header row.
The placeholder files that ship with this project contain only the header and one example row.
`run_real.py` refuses to run while it still sees the placeholder rows.

| File | What it holds | Columns / shape | Where it comes from |
|---|---|---|---|
| `drug_disease.csv` | known drug treats disease pairs | `drug_id,disease_id` | CTD (ctdbase.org), keep diseases with OMIM ids; or the Gottlieb et al. 2011 set |
| `drug_target.csv` | known drug binds target pairs | `drug_id,target_id` | DrugBank (approved small molecules, drugs with 2+ targets) |
| `drug_similarity.csv` | chemical similarity, all drugs | square matrix: first column = drug_id, header = drug_ids, values 0..1 | Tanimoto on 2D fingerprints from DrugBank SMILES (CDK or RDKit) |
| `disease_similarity.csv` | disease similarity | square matrix, same layout | MeSH semantic similarity (DisSim web server) |
| `target_similarity.csv` | protein similarity | square matrix, same layout | Smith-Waterman on sequences (gap open 11, extend 1, BLOSUM62), scaled to 0..1 |

IDs must match across files (the same drug needs the same `drug_id` everywhere).
Drugs that appear in both `drug_disease.csv` and `drug_target.csv` become the anchor links
(the paper had 469 of them on CTD + DrugBank).
