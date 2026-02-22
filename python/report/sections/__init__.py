from report.sections.approval_prediction import ApprovalPredictionSection
from report.sections.ghost_permits import GhostPermitsSection
from report.sections.green_belt import GreenBeltSection
from report.sections.income import IncomeProfileSection

# Registry — add new sections here in the order they should appear in the report.
SECTIONS = [
    IncomeProfileSection(),
    GhostPermitsSection(),
    GreenBeltSection(),
    ApprovalPredictionSection(),
]
