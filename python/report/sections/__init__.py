from report.sections.approval_prediction import ApprovalPredictionSection
from report.sections.income import IncomeProfileSection

# Registry — add new sections here in the order they should appear in the report.
SECTIONS = [
    IncomeProfileSection(),
    ApprovalPredictionSection(),
]
