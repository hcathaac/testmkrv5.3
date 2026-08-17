# Research Command Chair — operating guide

## Purpose

Module 12B converts a precisely selected spreadsheet and PDF evidence scope into a reproducible research protocol. It is an additive module: the econometric, GIS, clustering, MCDA, Monte Carlo, predictive and publication modules remain unchanged.

## Cost and privacy

The built-in interpreter is free and requires no API key. Spreadsheet and PDF contents are processed within the running application. A locally installed Ollama model may be selected for richer prose; Ollama is optional and is normally unavailable on Streamlit Community Cloud.

## Workflow

1. Upload XLSX/CSV files through the existing sidebar and choose the active dataset.
2. Open Module 12B and select no more than 1,000 retained variables.
3. Select an optional year/date variable and exact year interval.
4. Add up to six categorical or numeric row filters.
5. Upload one or more PDFs. Select documents, page intervals and optional keywords.
6. State one principal research question, the algorithm, equation, steps and limitations.
7. Optionally create a derived numeric variable with the safe expression engine.
8. Run the Research Command and inspect every generated table and warning.
9. Ask questions about the selected evidence.
10. Download the Word/Markdown paper blueprint and the complete reproducibility bundle.

## Safe equations

The LaTeX equation is preserved for display and manuscript reporting. A separate computable expression can create one derived variable. Allowed components are numeric column names, constants, parentheses, `+`, `-`, `*`, `/`, `**`, `%`, `log`, `log1p`, `exp`, `sqrt` and `abs`. Python statements, attributes, imports, file operations and system commands are rejected.

## Built-in algorithms

- **Descriptive profile:** analytical-sample audit and numeric summaries.
- **Longitudinal trend:** mean, sum, median or count by the selected year and optional group.
- **Correlation screening:** pairwise Spearman associations and p-values.
- **OLS specification:** OLS with HC3 heteroskedasticity-robust standard errors.
- **Custom documented algorithm:** records the supplied steps and limitations, but does not misrepresent prose as validated executable code.

Use the specialist analytical modules for panel FE/RE, DiD, IV/2SLS, spatial inference, Monte Carlo, clustering, prediction and MCDA after the Research Chair has defined the evidence scope and protocol.

## PDF limitations

Text-native PDFs are extracted page by page. Image-only scans require OCR before upload. Keyword matching is literal and does not establish conceptual relevance. Selected passages are research notes, not automatically verified quotations or complete references; always check the original page before submission.

## Paper blueprint

The generated blueprint includes contribution, research question, hypotheses, data scope, algorithm, equation, result tables, interpretation, robustness plan, limitations, recommended manuscript structure and selected PDF locations. It is a structured drafting aid, not an automatically publishable paper.

## Reproducibility bundle

The ZIP contains:

- filtered analytical data in CSV;
- selected PDF evidence in CSV and text;
- protocol JSON;
- every result table in CSV;
- consolidated XLSX workbook;
- paper blueprint in Markdown and Word;
- a README stating the scientific safeguards.
