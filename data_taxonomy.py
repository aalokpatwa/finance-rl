"""
Creates a logical taxonomy between financial metrics and their line item dependencies.
A "basic" metric has no dependencies and can come from raw data.
A "derived" metric has dependencies, and an associated function that computes the metric.
"""

from typing import Dict, List, Callable, Union, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

MetricFunction = Callable[..., float]
ReasoningFunction = Callable[..., str]

class MetricType(Enum):
    BASIC = auto()
    DERIVED = auto()

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class QuestionType(Enum):
    CALCULATE = "calculate"
    COMPARE = "compare"

@dataclass
class MetricDefinition:
    metric_type: MetricType
    required: bool = True
    formula: Optional[Union[MetricFunction, Callable[..., bool]]] = None
    level: DifficultyLevel = DifficultyLevel.EASY
    reasoning: Optional[ReasoningFunction] = None
    units: str = ""
    question_type: QuestionType = QuestionType.CALCULATE
    question_templates: List[str] = field(default_factory=list)

def check_if_present(data: dict, metric_name: str, include_string: str) -> str:
    """Check if a metric is present in the data and return the include string if it is."""
    if metric_name in data[list(data.keys())[0]]:
        return include_string
    return ""

class MetricCategory:
    """Base class for organizing metrics into logical categories."""
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {}

class BasicIncomeMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Revenue": MetricDefinition(metric_type=MetricType.BASIC),
            "Cost of Goods Sold": MetricDefinition(metric_type=MetricType.BASIC),
            "SG&A Expense": MetricDefinition(metric_type=MetricType.BASIC),
            "R&D Expense": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Depreciation Expense": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Stock Based Compensation Expense": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Interest Expense": MetricDefinition(metric_type=MetricType.BASIC),
            "Income Tax Expense": MetricDefinition(metric_type=MetricType.BASIC),
            "Tax Rate": MetricDefinition(metric_type=MetricType.BASIC, units="%"),
        }

class DerivedIncomeMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Gross Income": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: data[year]["Revenue"] - data[year]["Cost of Goods Sold"],
                level=DifficultyLevel.EASY,
                reasoning=lambda data, year: (
                    f"{year} Gross Income is calculated by subtracting {year} Cost of Goods Sold from {year} Revenue."
                    + get_metric_reasoning("Revenue", data, year)
                    + get_metric_reasoning("Cost of Goods Sold", data, year)
                ),
                question_templates=[
                    "Calculate Gross Income for {year}",
                    "What was the Gross Income in {year}?",
                    "Determine the company's Gross Income for fiscal year {year}",
                    "Find the total Gross Income generated in {year}",
                    "Compute the Gross Income figure for {year}"
                ]
            ),
            "EBITDA": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    data[year]["Revenue"]
                    - data[year]["Cost of Goods Sold"]
                    - data[year]["SG&A Expense"]
                    - data[year].get("R&D Expense", 0)
                    - data[year].get("Stock Based Compensation Expense", 0)
                ),
                level=DifficultyLevel.EASY,
                reasoning=lambda data, year: "".join([
                    f"{year} EBITDA is calculated by subtracting ",
                    f"{year} Cost of Goods Sold, {year} SG&A Expense, ",
                    check_if_present(data, "R&D Expense", f"{year} R&D Expense, "),
                    check_if_present(data, "Stock Based Compensation Expense", f"{year} Stock Based Compensation Expense, "),
                    f"from {year} Revenue.",
                    get_metric_reasoning("Revenue", data, year),
                    get_metric_reasoning("Cost of Goods Sold", data, year),
                    get_metric_reasoning("SG&A Expense", data, year),
                    check_if_present(data, "R&D Expense", get_metric_reasoning("R&D Expense", data, year)),
                    check_if_present(data, "Stock Based Compensation Expense", get_metric_reasoning("Stock Based Compensation Expense", data, year))
                ]),
                question_templates=[
                    "Calculate EBITDA for {year}",
                    "What was the company's EBITDA in {year}?",
                    "Determine the EBITDA value for fiscal year {year}",
                    "Find the Earnings Before Interest, Taxes, Depreciation, and Amortization (EBITDA) for {year}",
                    "Compute the total EBITDA figure for {year}"
                ]
            ),
            "Operating Income": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("EBITDA", data, year) 
                    - data[year].get("Depreciation Expense", 0)
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"Operating Income for {year} is calculated by subtracting Depreciation Expense from EBITDA."
                    + get_metric_reasoning("EBITDA", data, year)
                    + check_if_present(data, "Depreciation Expense", get_metric_reasoning("Depreciation Expense", data, year))
                ),
                question_templates=[
                    "Calculate Operating Income for {year}",
                    "What was the Operating Income in {year}?",
                    "Determine the company's Operating Income for fiscal year {year}",
                    "Find the total Operating Income generated in {year}",
                    "Compute the Operating Income figure for {year}"
                ]
            ),
            "NOPAT": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: get_metric("Operating Income", data, year) * (1 - data[year]["Tax Rate"] / 100),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"{year} NOPAT is calculated by multiplying Operating Income by (1 - Tax Rate). "
                    + get_metric_reasoning("Operating Income", data, year)
                    + get_metric_reasoning("Tax Rate", data, year)
                ),
                question_templates=[
                    "Calculate Net Operating Profit After Taxes (NOPAT) for {year}",
                    "What was the company's NOPAT in {year}?",
                    "Determine the NOPAT value for fiscal year {year}",
                    "Find the Net Operating Profit After Taxes (NOPAT) for {year}",
                    "Compute the total NOPAT figure for {year}"
                ]
            ),
        }

class GrowthMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Revenue Growth": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, start_year, end_year: (
                    (data[end_year]["Revenue"] - data[start_year]["Revenue"]) 
                    / data[start_year]["Revenue"] * 100
                ),
                level=DifficultyLevel.EASY,
                reasoning=lambda data, start_year, end_year: (
                    f"Revenue Growth from {start_year} to {end_year} is calculated as:\n"
                    f"({end_year} Revenue - {start_year} Revenue) / {start_year} Revenue * 100"
                    + get_metric_reasoning("Revenue", data, start_year)
                    + get_metric_reasoning("Revenue", data, end_year)
                ),
                units="%",
                question_templates=[
                    "Calculate Revenue Growth from {start_year} to {end_year}",
                    "What was the percentage growth in Revenue between {start_year} and {end_year}?",
                    "By what percentage did Revenue increase from {start_year} to {end_year}?",
                    "Determine the Revenue growth rate from {start_year} to {end_year}",
                    "How much did Revenue grow (as a percentage) between {start_year} and {end_year}?"
                ]
            ),
            "Operating Margin": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Operating Income", data, year) / data[year]["Revenue"] * 100
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"Operating Margin for {year} is calculated as:\n"
                    f"Operating Income / Revenue * 100"
                    + get_metric_reasoning("Operating Income", data, year)
                    + get_metric_reasoning("Revenue", data, year)
                ),
                units="%",
                question_templates=[
                    "Calculate Operating Margin for {year}",
                    "What percentage of Revenue was Operating Income in {year}?",
                    "Determine the Operating Margin percentage for {year}",
                    "Find the Operating Margin (as a percentage of Revenue) in {year}",
                    "What was the company's Operating Margin in {year}?"
                ]
            ),
        }

class BalanceSheetMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Accounts Payable": MetricDefinition(metric_type=MetricType.BASIC),
            "Accrued Salaries": MetricDefinition(metric_type=MetricType.BASIC),
            "Deferred Revenue": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Current Portion of Long-Term Debt": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Long-term Debt": MetricDefinition(metric_type=MetricType.BASIC),
            "Working Cash": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: max(
                    min(data[year].get("Cash", 0) + data[year].get("Marketable Securities", 0), 0.02 * data[year]["Revenue"]),
                    0
                ),
                level=DifficultyLevel.EASY,
                reasoning=lambda data, year: (
                    f"Working Cash is calculated as the minimum of Cash {check_if_present(data, 'Marketable Securities', 'plus Marketable Securities')} and 2% of Revenue. "
                    + get_metric_reasoning("Cash", data, year)
                    + check_if_present(data, "Marketable Securities", get_metric_reasoning("Marketable Securities", data, year))
                    + get_metric_reasoning("Revenue", data, year)
                ),
                question_templates=[
                    "Calculate Working Cash for {year}",
                    "What was the Working Cash in {year}?",
                    "Determine the Working Cash value for fiscal year {year}",
                    "Find the Working Cash figure for {year}",
                    "Compute the total Working Cash for {year}"
                ]
            ),
        }

class AssetMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Cash": MetricDefinition(metric_type=MetricType.BASIC),
            "Marketable Securities": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Inventory": MetricDefinition(metric_type=MetricType.BASIC),
            "Accounts Receivable": MetricDefinition(metric_type=MetricType.BASIC),
            "Prepaid Assets": MetricDefinition(metric_type=MetricType.BASIC, required=False),
            "Property and Equipment": MetricDefinition(metric_type=MetricType.BASIC),
            "Intangible Assets": MetricDefinition(metric_type=MetricType.BASIC),
            "Other Assets": MetricDefinition(metric_type=MetricType.BASIC),
        }

class WorkingCapitalMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Operating Current Assets": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Working Cash", data, year) +
                    get_metric("Inventory", data, year) +
                    get_metric("Accounts Receivable", data, year) +
                    get_metric("Prepaid Assets", data, year)
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"{year} Operating Current Assets is calculated by adding Working Cash, "
                    f"Inventory, Accounts Receivable"
                    f"{check_if_present(data, 'Prepaid Assets', ', and Prepaid Assets')}. "
                    + get_metric_reasoning("Working Cash", data, year)
                    + get_metric_reasoning("Inventory", data, year)
                    + get_metric_reasoning("Accounts Receivable", data, year)
                    + check_if_present(data, "Prepaid Assets", get_metric_reasoning("Prepaid Assets", data, year))
                ),
                question_templates=[
                    "Calculate Operating Current Assets for {year}",
                    "What was the Operating Current Assets in {year}?",
                    "Determine the Operating Current Assets value for fiscal year {year}",
                    "Find the Operating Current Assets figure for {year}",
                    "Compute the total Operating Current Assets for {year}"
                ]
            ),
            "Operating Current Liabilities": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Accounts Payable", data, year) +
                    get_metric("Accrued Salaries", data, year) +
                    get_metric("Deferred Revenue", data, year)
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"{year} Operating Current Liabilities is calculated by adding Accounts Payable, "
                    f"Accrued Salaries{check_if_present(data, 'Deferred Revenue', ', and Deferred Revenue')}. "
                    + get_metric_reasoning("Accounts Payable", data, year)
                    + get_metric_reasoning("Accrued Salaries", data, year)
                    + check_if_present(data, "Deferred Revenue", get_metric_reasoning("Deferred Revenue", data, year))
                ),
                question_templates=[
                    "Calculate Operating Current Liabilities for {year}",
                    "What was the Operating Current Liabilities in {year}?",
                    "Determine the Operating Current Liabilities value for fiscal year {year}",
                    "Find the Operating Current Liabilities figure for {year}",
                    "Compute the total Operating Current Liabilities for {year}"
                ]
            ),
            "Net Working Capital": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Operating Current Assets", data, year) -
                    get_metric("Operating Current Liabilities", data, year)
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"{year} Net Working Capital is calculated by subtracting Operating Current Liabilities "
                    f"from Operating Current Assets. "
                    + get_metric_reasoning("Operating Current Assets", data, year)
                    + get_metric_reasoning("Operating Current Liabilities", data, year)
                ),
                question_templates=[
                    "Calculate Net Working Capital for {year}",
                    "What was the Net Working Capital in {year}?",
                    "Determine the Net Working Capital value for fiscal year {year}",
                    "Find the Net Working Capital figure for {year}",
                    "Compute the total Net Working Capital for {year}"
                ]
            ),
        }

class PerformanceMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Invested Capital": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Net Working Capital", data, year) +
                    get_metric("Property and Equipment", data, year) +
                    get_metric("Intangible Assets", data, year) +
                    get_metric("Other Assets", data, year)
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, year: (
                    f"{year} Invested Capital is calculated by adding Net Working Capital, "
                    f"Property and Equipment, Intangible Assets, and Other Assets. "
                    + get_metric_reasoning("Net Working Capital", data, year)
                    + get_metric_reasoning("Property and Equipment", data, year)
                    + get_metric_reasoning("Intangible Assets", data, year)
                    + get_metric_reasoning("Other Assets", data, year)
                ),
                question_templates=[
                    "Calculate Invested Capital for {year}",
                    "What was the Invested Capital in {year}?",
                    "Determine the Invested Capital value for fiscal year {year}",
                    "Find the Invested Capital figure for {year}",
                    "Compute the total Invested Capital for {year}"
                ]
            ),
            "Capital Turnover": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Revenue", data, year) /
                    get_metric("Invested Capital", data, year)
                ),
                level=DifficultyLevel.HARD,
                reasoning=lambda data, year: (
                    f"{year} Capital Turnover is calculated by dividing Revenue by Invested Capital."
                    + get_metric_reasoning("Revenue", data, year)
                    + get_metric_reasoning("Invested Capital", data, year)
                ),
                units="x",
                question_templates=[
                    "Calculate Capital Turnover for {year}",
                    "What was the Capital Turnover in {year}?",
                    "Determine the Capital Turnover value for fiscal year {year}",
                    "Find the Capital Turnover figure for {year}",
                    "Compute the total Capital Turnover for {year}"
                ]
            ),
            "Return on Invested Capital": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("NOPAT", data, year) /
                    get_metric("Invested Capital", data, year) * 100
                ),
                level=DifficultyLevel.HARD,
                reasoning=lambda data, year: (
                    f"{year} Return on Invested Capital is calculated by dividing NOPAT by Invested Capital."
                    + get_metric_reasoning("NOPAT", data, year)
                    + get_metric_reasoning("Invested Capital", data, year)
                ),
                units="%",
                question_templates=[
                    "Calculate Return on Invested Capital for {year}",
                    "What was the Return on Invested Capital in {year}?",
                    "Determine the Return on Invested Capital value for fiscal year {year}",
                    "Find the Return on Invested Capital figure for {year}",
                    "Compute the total Return on Invested Capital for {year}"
                ]
            ),
            "Current Ratio": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    (get_metric("Cash", data, year) +
                     get_metric("Marketable Securities", data, year) +
                     get_metric("Accounts Receivable", data, year) +
                     get_metric("Prepaid Assets", data, year) +
                     get_metric("Inventory", data, year)) /
                    (get_metric("Accounts Payable", data, year) +
                     get_metric("Accrued Salaries", data, year) +
                     get_metric("Deferred Revenue", data, year) +
                     get_metric("Current Portion of Long-Term Debt", data, year))
                ),
                level=DifficultyLevel.HARD,
                reasoning=lambda data, year: (
                    f"{year} Current Ratio is calculated by dividing current assets "
                    f"(Cash{check_if_present(data, 'Marketable Securities', ', Marketable Securities')}, "
                    f"Accounts Receivable, Inventory{check_if_present(data, 'Prepaid Assets', ', and Prepaid Assets')}) "
                    f"by current liabilities (Accounts Payable, Accrued Salaries"
                    f"{check_if_present(data, 'Deferred Revenue', ', Deferred Revenue')}"
                    f"{check_if_present(data, 'Current Portion of Long-Term Debt', ', and Current Portion of Long-Term Debt')}). "
                    + get_metric_reasoning("Cash", data, year)
                    + check_if_present(data, "Marketable Securities", get_metric_reasoning("Marketable Securities", data, year))
                    + get_metric_reasoning("Accounts Receivable", data, year)
                    + get_metric_reasoning("Inventory", data, year)
                    + check_if_present(data, "Prepaid Assets", get_metric_reasoning("Prepaid Assets", data, year))
                    + get_metric_reasoning("Accounts Payable", data, year)
                    + get_metric_reasoning("Accrued Salaries", data, year)
                    + check_if_present(data, "Deferred Revenue", get_metric_reasoning("Deferred Revenue", data, year))
                    + check_if_present(data, "Current Portion of Long-Term Debt", get_metric_reasoning("Current Portion of Long-Term Debt", data, year))
                ),
                units="x",
                question_templates=[
                    "Calculate Current Ratio for {year}",
                    "What was the Current Ratio in {year}?",
                    "Determine the Current Ratio value for fiscal year {year}",
                    "Find the Current Ratio figure for {year}",
                    "Compute the total Current Ratio for {year}"
                ]
            ),
            "Interest Coverage": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, year: (
                    get_metric("Operating Income", data, year) / get_metric("Interest Expense", data, year)
                ),
                level=DifficultyLevel.HARD,
                reasoning=lambda data, year: (
                    f"{year} Interest Coverage is calculated by dividing Operating Income by Interest Expense. "
                    + get_metric_reasoning("Operating Income", data, year)
                    + get_metric_reasoning("Interest Expense", data, year)
                ),
                units="x",
                question_templates=[
                    "Calculate Interest Coverage for {year}",
                    "What was the Interest Coverage in {year}?"
                ]
            ) 
        }
        
class ComparativeMetrics(MetricCategory):
    @staticmethod
    def get_definitions() -> Dict[str, MetricDefinition]:
        return {
            "Is R&D growth faster than Revenue growth from {start_year} to {end_year}": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, start_year, end_year: (
                    # Calculate R&D growth
                    "Yes" if ((data[end_year].get("R&D Expense", 0) - data[start_year].get("R&D Expense", 0)) 
                     / data[start_year].get("R&D Expense", 1)) >
                    # Compare with Revenue growth
                    ((data[end_year]["Revenue"] - data[start_year]["Revenue"]) 
                     / data[start_year]["Revenue"]) else "No"
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, start_year, end_year: (
                    f"Let's compare R&D and Revenue growth from {start_year} to {end_year}:\n"
                    + get_metric_reasoning("R&D Expense", data, start_year)
                    + get_metric_reasoning("R&D Expense", data, end_year)
                    + get_metric_reasoning("Revenue Growth", data, start_year, end_year)
                ),
                units="",
                question_type=QuestionType.COMPARE,
                question_templates=[
                    "Are R&D investments growing faster than revenue from {start_year} to {end_year}?",
                    "Compare R&D growth and Revenue growth from {start_year} to {end_year}.",
                    "Determine if R&D investments are growing faster than revenue from {start_year} to {end_year}.",
                    "Find out if R&D growth is higher than Revenue growth from {start_year} to {end_year}.",
                    "Analyze the growth of R&D investments and revenue from {start_year} to {end_year}."
                ]
            ),
            "Has Operating Margin expanded": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, start_year, end_year: (
                    "Yes" if (get_metric("Operating Income", data, end_year) / get_metric("Revenue", data, end_year)) >
                            (get_metric("Operating Income", data, start_year) / get_metric("Revenue", data, start_year)) 
                    else "No"
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, start_year, end_year: (
                    f"To determine if Operating Margin is expanding, let's compare Operating Margin from {start_year} to {end_year}:\n"
                    + get_metric_reasoning("Operating Margin", data, start_year)
                    + get_metric_reasoning("Operating Margin", data, end_year)
                ),
                question_type=QuestionType.COMPARE,
                question_templates=[
                    "Is the Operating Margin expanding from {start_year} to {end_year}?",
                    "Has the Operating Margin improved from {start_year} to {end_year}?",
                    "Compare Operating Margins between {start_year} and {end_year}.",
                    "Determine if Operating Margin shows expansion from {start_year} to {end_year}.",
                    "Analyze Operating Margin trend from {start_year} to {end_year}."
                ]
            ),
            "Has Working Capital improved": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, start_year, end_year: (
                    "Yes" if get_metric("Net Working Capital", data, end_year) >
                            get_metric("Net Working Capital", data, start_year)
                    else "No"
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, start_year, end_year: (
                    f"To determine if Working Capital is improving, let's compare Net Working Capital from {start_year} to {end_year}:\n"
                    + get_metric_reasoning("Net Working Capital", data, start_year)
                    + get_metric_reasoning("Net Working Capital", data, end_year)
                ),
                question_type=QuestionType.COMPARE,
                question_templates=[
                    "Has Working Capital improved from {start_year} to {end_year}?",
                    "Is Net Working Capital higher in {end_year} compared to {start_year}?",
                    "Compare Working Capital between {start_year} and {end_year}.",
                    "Determine if Working Capital shows improvement from {start_year} to {end_year}.",
                    "Analyze Working Capital trend from {start_year} to {end_year}."
                ]
            ),
            "Has Interest Coverage improved": MetricDefinition(
                metric_type=MetricType.DERIVED,
                formula=lambda data, start_year, end_year: (
                    "Yes" if (get_metric("Operating Income", data, end_year) / get_metric("Interest Expense", data, end_year)) >
                            (get_metric("Operating Income", data, start_year) / get_metric("Interest Expense", data, start_year))
                    else "No"
                ),
                level=DifficultyLevel.MEDIUM,
                reasoning=lambda data, start_year, end_year: (
                    f"To determine if Interest Coverage is improving, let's compare {start_year} and {end_year} Interest Coverage:\n"
                    + get_metric_reasoning("Interest Coverage", data, start_year)
                    + get_metric_reasoning("Interest Coverage", data, end_year)
                ),
                question_type=QuestionType.COMPARE,
                question_templates=[
                    "Has the Interest Coverage Ratio improved from {start_year} to {end_year}?",
                    "Is the company's ability to cover interest payments better in {end_year} vs {start_year}?",
                    "Compare Interest Coverage Ratios between {start_year} and {end_year}.",
                    "Determine if Interest Coverage shows improvement from {start_year} to {end_year}.",
                    "Analyze Interest Coverage trend from {start_year} to {end_year}."
                ]
            )
        }

# Combine all metric definitions
METRIC_DEFINITIONS: Dict[str, MetricDefinition] = {
    **BasicIncomeMetrics.get_definitions(),
    **DerivedIncomeMetrics.get_definitions(),
    **GrowthMetrics.get_definitions(),
    **BalanceSheetMetrics.get_definitions(),
    **AssetMetrics.get_definitions(),
    **WorkingCapitalMetrics.get_definitions(),
    **PerformanceMetrics.get_definitions(),
    **ComparativeMetrics.get_definitions(),
}

def get_metric(metric_name: str, data: dict, year: Union[int, str], *args) -> Union[str, float]:
    """
    Get the value of a metric for a given year.
    - If 'metric_name' is a basic metric, returns data[year][metric_name].
    - If it's derived, recursively compute dependencies then run its formula.
    - Additional years can be passed in `args` if the formula needs them (e.g. for 'growth').
    """
    if metric_name not in METRIC_DEFINITIONS:
        return 0
    
    definition = METRIC_DEFINITIONS[metric_name]
    
    if definition.metric_type == MetricType.BASIC and metric_name not in data[year]:
        return 0 # The data type was not required and not present in the data
    
    if definition.metric_type == MetricType.BASIC:
        return data[year][metric_name]
    
    return definition.formula(data, year, *args)

def get_formatted_metric(metric_name: str, data: dict, year: Union[int, str], *args) -> str:
    metric_value: Union[str, float] = get_metric(metric_name, data, year, *args)
    
    
    if isinstance(metric_value, float) or isinstance(metric_value, int):
        metric_value = str(round(metric_value, 1))
    
    return metric_value

def get_metric_reasoning(metric_name: str, data: dict, year: Union[int, str], *args) -> str:
    """Get the reasoning for how a metric was calculated."""
    definition = METRIC_DEFINITIONS[metric_name]
    
    if definition.reasoning is None or definition.metric_type == MetricType.BASIC:
        return f"\n{year} {metric_name} is {get_metric(metric_name, data, year, *args)}{definition.units}."
    
    return "\n" + definition.reasoning(data, year, *args) + f"\nTherefore, {metric_name} is {get_formatted_metric(metric_name, data, year, *args)}{definition.units}."

def get_all_derived_metrics() -> List[Tuple[str, DifficultyLevel]]:
    """Returns a list of all the names and difficulty levels of derived metrics."""
    return [
        (metric_name, definition.level) for metric_name, definition in METRIC_DEFINITIONS.items()
        if definition.metric_type == MetricType.DERIVED
    ]

def get_all_basic_metrics() -> List[Tuple[str, DifficultyLevel, bool]]:
    """Returns a list of all basic metrics and their difficulty levels."""
    return [
        (metric_name, definition.level, definition.required) for metric_name, definition in METRIC_DEFINITIONS.items()
        if definition.metric_type == MetricType.BASIC
    ]

def get_number_of_years_required(metric_name: str) -> int:
    """Returns the number of years required to calculate a metric."""
    definition = METRIC_DEFINITIONS[metric_name]
    return definition.formula.__code__.co_argcount - 1

if __name__ == "__main__":
    # Create some fake data
    sample_data = {
        2021: {
            "Revenue": 2000,
            "Cost of Goods Sold": 800,
            "SG&A Expense": 400,
            "R&D Expense": 200,
            "Depreciation Expense": 100,
            "Interest Expense": 50,
            "Income Tax Expense": 100,
            "Tax Rate": 25,
        },
        2022: {
            "Revenue": 2560,
            "Cost of Goods Sold": 846,
            "SG&A Expense": 405,
            "R&D Expense": 768,
            "Depreciation Expense": 283,
            "Interest Expense": 110,
            "Income Tax Expense": 209,
            "Tax Rate": 28,
        }
    }

    # Test derived metrics
    metrics_to_test = [
        ("Operating Income", 2022),
        ("EBITDA", 2022),
        ("NOPAT", 2022),
        ("Revenue Growth", 2021, 2022),
    ]

    for metric_test in metrics_to_test:
        metric_name = metric_test[0]
        years = metric_test[1:]
        result = get_metric(metric_name, sample_data, *years)
        print(f"\n{metric_name} ({', '.join(map(str, years))}): {result}")
        print(get_metric_reasoning(metric_name, sample_data, *years))
