import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath, previewDir, verificationPath] = process.argv.slice(2);
const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const passStatuses = [...new Set(payload.matrix.map((row) => String(row.status ?? "")).filter((status) => status.startsWith("PASS")))].sort();
const passFormula = passStatuses.length
  ? "=" + passStatuses.map((status) => "COUNTIF('Validation Matrix'!$N$2:$N$" + (payload.matrix.length + 1) + ",\"" + status.replaceAll('"', '""') + "\")").join("+")
  : "=0";
const expectedSummary = {
  validation_rows: payload.matrix.length,
  passed: payload.matrix.filter((row) => String(row.status ?? "").startsWith("PASS")).length,
  inconclusive: payload.matrix.filter((row) => row.status === "INCONCLUSIVE").length,
  expected_failures: payload.matrix.filter((row) => row.status === "FAIL_EXPECTED").length,
  blockers: payload.matrix.filter((row) => ["FAIL_THEORY", "FAIL_IMPLEMENTATION"].includes(row.status)).length,
  error_budget_observables: payload.error_budget.length,
  unknown_error_components: payload.error_budget.reduce((total, row) => total + Number(row.unknown_error_component_count ?? 0), 0),
};
const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const matrix = workbook.worksheets.add("Validation Matrix");
const budget = workbook.worksheets.add("Error Budget");
const provenance = workbook.worksheets.add("Provenance");

const navy = "#17324D";
const blue = "#D9EAF7";
const pale = "#F4F7FA";
const green = "#DDF2E2";
const amber = "#FFF0C2";
const red = "#F8D7DA";

function writeTable(sheet, rows, columns, tableName) {
  const values = [columns, ...rows.map((row) => columns.map((column) => row[column] ?? null))];
  const target = sheet.getRangeByIndexes(0, 0, values.length, columns.length);
  target.values = values;
  sheet.getRangeByIndexes(0, 0, 1, columns.length).format = {
    fill: navy, font: { bold: true, color: "#FFFFFF" },
    wrapText: true, verticalAlignment: "center",
  };
  sheet.getRangeByIndexes(1, 0, Math.max(1, rows.length), columns.length).format = {
    fill: "#FFFFFF", verticalAlignment: "top",
    borders: { preset: "inside", style: "thin", color: "#E3E8ED" },
  };
  if (rows.length > 0) {
    const table = sheet.tables.add(target, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
  }
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  target.format.wrapText = true;
  target.format.autofitRows();
  return target;
}

const matrixColumns = [
  "theorem_id", "claim_name", "claim_layer", "model_level", "code_id", "run_id",
  "validation_type", "parameter_set", "residual_value", "tolerance",
  "certified_lower_bound", "certified_upper_bound", "physical_margin", "status",
  "raw_data_file", "derived_data_file", "certificate_file", "future_figure_id", "notes",
];
const budgetColumns = [
  "observable_id", "task_id", "model_scope", "status", "truncation_error", "physical_tail",
  "cover_error", "angular_error", "solver_error", "floating_point_interval_width",
  "transport_error", "representation_error", "dos_smoothing_error",
  "statistical_kpm_slq_error", "known_error_component_count", "unknown_error_component_count",
  "total_known_error_upper", "notes",
];
const provenanceColumns = [
  "name", "url", "revision", "doi", "retrieved_at_utc", "tree_sha256", "status", "scope",
];
writeTable(matrix, payload.matrix, matrixColumns, "ValidationMatrixTable");
writeTable(budget, payload.error_budget, budgetColumns, "ErrorBudgetTable");
writeTable(provenance, payload.provenance, provenanceColumns, "ProvenanceTable");

summary.showGridLines = false;
summary.getRange("A1:H1").merge();
summary.getRange("A1").values = [["Computer-Assisted Validation — Phase D Audit"]];
summary.getRange("A1:H1").format = {
  fill: navy, font: { bold: true, color: "#FFFFFF", size: 16 },
  rowHeight: 30, verticalAlignment: "center",
};
summary.getRange("A3:B3").values = [["Metric", "Value"]];
summary.getRange("A3:B3").format = { fill: blue, font: { bold: true, color: navy } };
summary.getRange("A4:A11").values = [
  ["Run ID"], ["Validation rows"], ["Passed / certified / external"], ["Inconclusive"],
  ["Expected failures"], ["Theory/implementation blockers"], ["Error-budget observables"], ["Unknown error components"],
];
summary.getRange("B4").values = [[payload.run_id]];
const lastMatrixRow = payload.matrix.length + 1;
const lastBudgetRow = payload.error_budget.length + 1;
summary.getRange("B5:B11").formulas = [
  [`=COUNTA('Validation Matrix'!$E$2:$E$${lastMatrixRow})`],
  [passFormula],
  [`=COUNTIF('Validation Matrix'!$N$2:$N$${lastMatrixRow},"INCONCLUSIVE")`],
  [`=COUNTIF('Validation Matrix'!$N$2:$N$${lastMatrixRow},"FAIL_EXPECTED")`],
  [`=COUNTIF('Validation Matrix'!$N$2:$N$${lastMatrixRow},"FAIL_THEORY")+COUNTIF('Validation Matrix'!$N$2:$N$${lastMatrixRow},"FAIL_IMPLEMENTATION")`],
  [`=COUNTA('Error Budget'!$A$2:$A$${lastBudgetRow})`],
  [`=SUM('Error Budget'!$P$2:$P$${lastBudgetRow})`],
];
summary.getRange("A4:B11").format = {
  borders: { preset: "outside", style: "thin", color: "#A9B7C4" },
  verticalAlignment: "center",
};
summary.getRange("A4:A11").format.fill = pale;
summary.getRange("A4:A11").format.font = { bold: true, color: navy };
summary.getRange("A13:H16").merge();
summary.getRange("A13").values = [[
  "Interpretation guard: finite-cover atomic spectra are not presented as an unsmoothed density theorem. " +
  "INCONCLUSIVE and FAIL_EXPECTED entries remain visible and are not counted as passes."
]];
summary.getRange("A13:H16").format = { fill: amber, wrapText: true, verticalAlignment: "center" };
summary.getRange("A1:H16").format.font.name = "Aptos";
summary.getRange("A1:A16").format.columnWidth = 34;
summary.getRange("B1:B16").format.columnWidth = 72;
summary.freezePanes.freezeRows(1);

matrix.getRange(`N2:N${lastMatrixRow}`).conditionalFormats.add("containsText", { text: "PASS", format: { fill: green, font: { color: "#176B2C" } } });
matrix.getRange(`N2:N${lastMatrixRow}`).conditionalFormats.add("containsText", { text: "INCONCLUSIVE", format: { fill: amber, font: { color: "#7A5200" } } });
matrix.getRange(`N2:N${lastMatrixRow}`).conditionalFormats.add("containsText", { text: "FAIL", format: { fill: red, font: { color: "#8A1F28" } } });
matrix.getRange(`A1:S${lastMatrixRow}`).format.font.name = "Aptos";
matrix.getRange(`A1:A${lastMatrixRow}`).format.columnWidth = 22;
matrix.getRange(`B1:B${lastMatrixRow}`).format.columnWidth = 34;
matrix.getRange(`C1:D${lastMatrixRow}`).format.columnWidth = 22;
matrix.getRange(`E1:E${lastMatrixRow}`).format.columnWidth = 12;
matrix.getRange(`F1:F${lastMatrixRow}`).format.columnWidth = 20;
matrix.getRange(`G1:N${lastMatrixRow}`).format.columnWidth = 16;
matrix.getRange(`O1:Q${lastMatrixRow}`).format.columnWidth = 45;
matrix.getRange(`R1:R${lastMatrixRow}`).format.columnWidth = 16;
matrix.getRange(`S1:S${lastMatrixRow}`).format.columnWidth = 38;
budget.getRange(`A1:R${lastBudgetRow}`).format.font.name = "Aptos";
budget.getRange(`A1:D${lastBudgetRow}`).format.columnWidth = 26;
budget.getRange(`E1:Q${lastBudgetRow}`).format.columnWidth = 16;
budget.getRange(`R1:R${lastBudgetRow}`).format.columnWidth = 42;
const lastProvenanceRow = payload.provenance.length + 1;
provenance.getRange(`A1:H${lastProvenanceRow}`).format.font.name = "Aptos";
provenance.getRange(`A1:A${lastProvenanceRow}`).format.columnWidth = 22;
provenance.getRange(`B1:B${lastProvenanceRow}`).format.columnWidth = 45;
provenance.getRange(`C1:F${lastProvenanceRow}`).format.columnWidth = 24;
provenance.getRange(`G1:H${lastProvenanceRow}`).format.columnWidth = 34;
if (lastProvenanceRow >= 2) {
  provenance.getRange(`E2:E${lastProvenanceRow}`).format.numberFormat = "yyyy-mm-dd hh:mm:ss";
}

await fs.mkdir(previewDir, { recursive: true });
const previewRanges = {
  "Summary": "A1:H16",
  "Validation Matrix": "A1:S14",
  "Error Budget": "A1:R14",
  "Provenance": "A1:H8",
};
const rendered = [];
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const file = path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`);
  await fs.writeFile(file, new Uint8Array(await preview.arrayBuffer()));
  rendered.push(file);
}
const summaryInspection = await workbook.inspect({ kind: "table", range: "Summary!A1:H16", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 10 });
const errorInspection = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan" });
const formulaError = /(?:"value"|"text"|"displayValue")\s*:\s*"#(?:REF!|DIV\/0!|VALUE!|NAME\?|N\/A)"/;
const errorLines = String(errorInspection.ndjson ?? "")
  .split("\n")
  .filter((line) => formulaError.test(line));
const summaryValues = summary.getRange("B5:B11").values.flat().map((value) => Number(value));
const actualSummary = {
  validation_rows: summaryValues[0],
  passed: summaryValues[1],
  inconclusive: summaryValues[2],
  expected_failures: summaryValues[3],
  blockers: summaryValues[4],
  error_budget_observables: summaryValues[5],
  unknown_error_components: summaryValues[6],
};
const summaryReconciled = Object.keys(expectedSummary).every(
  (key) => Number.isFinite(actualSummary[key]) && Math.abs(actualSummary[key] - expectedSummary[key]) <= 1e-9,
);
const output = await SpreadsheetFile.exportXlsx(workbook);
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await output.save(outputPath);
await fs.writeFile(verificationPath, JSON.stringify({
  output: outputPath,
  rendered_sheet_count: rendered.length,
  rendered_sheets: rendered,
  rendered_ranges: previewRanges,
  formula_error_count: errorLines.length,
  summary_inspection: String(summaryInspection.ndjson ?? "").slice(0, 12000),
  error_scan: String(errorInspection.ndjson ?? "").slice(0, 12000),
  pass_statuses: passStatuses,
  summary_expected: expectedSummary,
  summary_actual: actualSummary,
  summary_reconciled: summaryReconciled,
  provenance_date_number_format: "yyyy-mm-dd hh:mm:ss",
}, null, 2));
console.log(JSON.stringify({ output: outputPath, rendered_sheet_count: rendered.length, formula_error_count: errorLines.length, summary_reconciled: summaryReconciled }));
