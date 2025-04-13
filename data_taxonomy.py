"""
Creates a logical taxonomy between financial metrics and their line item dependencies.
A “basic” metric has no dependencies and can come from raw data.
A “derived” metric has dependencies, and an associated function that computes the metric.
"""

from typing import Dict, List, Callable, Union

MetricFunction = Callable[..., float]

class MetricDefinition:
    def __init__(
        self, 
        metric_type: str,
        required: bool = True, 
        formula: Union[MetricFunction, None] = None, 
    ):
        self.metric_type = metric_type
        self.required = required
        self.formula = formula

# Hardcoded taxonomy for how line items should tie together
METRIC_DEFINITIONS: Dict[str, MetricDefinition] = {
    
    # Basic metrics - income statement
    "Revenue": MetricDefinition(metric_type="basic"),
    "Cost of Goods Sold": MetricDefinition(metric_type="basic"),
    "SG&A Expense": MetricDefinition(metric_type="basic"),
    "R&D Expense": MetricDefinition(metric_type="basic"),
    "Depreciation Expense": MetricDefinition(metric_type="basic"),
    "Interest Expense": MetricDefinition(metric_type="basic"),
    "Income Tax Expense": MetricDefinition(metric_type="basic"),
    "Tax Rate": MetricDefinition(metric_type="basic"),
    
    # Derived income statement metrics
    "EBITDA": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: data[year]["Revenue"] - data[year]["Cost of Goods Sold"] - data[year]["SG&A Expense"] - data[year]["R&D Expense"],
    ),
    "Operating Income": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: data[year]["Revenue"] - data[year]["Cost of Goods Sold"] - data[year]["SG&A Expense"] - data[year]["R&D Expense"] - data[year]["Depreciation Expense"],
    ),
    "NOPAT": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: get_metric("Operating Income", data, year) * (1 - data[year]["Tax Rate"]),
    ),
    "Net Income": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: get_metric("Operating Income", data, year) - data[year]["Income Tax Expense"] - data[year]["Interest Expense"],
    ),
    
    # Margin metrics
    "Gross Margin": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (data[year]["Revenue"] - data[year]["Cost of Goods Sold"]) / data[year]["Revenue"] * 100,
    ),
    "Operating Margin": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: get_metric("Operating Income", data, year) / data[year]["Revenue"] * 100,
    ),
    "Net Margin": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: get_metric("Net Income", data, year) / data[year]["Revenue"] * 100,
    ),
    
    # Growth-related metrics
    "Absolute Change in Net Margin": MetricDefinition(
        metric_type="derived",
        # This formula requires two years, so we can define a small wrapper:
        formula=lambda data, year1, year2: (
            get_metric("Net Margin", data, year2) 
            - get_metric("Net Margin", data, year1)
        ),
    ),
    
    # Basic metrics - balance sheet assets
    "Cash": MetricDefinition(metric_type="basic"),
    "Marketable Securities": MetricDefinition(metric_type="basic", required=False),
    "Inventory": MetricDefinition(metric_type="basic"),
    "Accounts Receivable": MetricDefinition(metric_type="basic"),
    "Prepaid Assets": MetricDefinition(metric_type="basic", required=False),
    "Property and Equipment": MetricDefinition(metric_type="basic"),
    "Intangible Assets": MetricDefinition(metric_type="basic", required=False),
    "Other Assets": MetricDefinition(metric_type="basic"),
    
    # Basic metrics - balance sheet liabilities
    "Accounts Payable": MetricDefinition(metric_type="basic"),
    "Accrued Salaries": MetricDefinition(metric_type="basic"),
    "Deferred Revenue": MetricDefinition(metric_type="basic"),
    "Current Portion of Long-Term Debt": MetricDefinition(metric_type="basic", required=False),
    "Long-term Debt": MetricDefinition(metric_type="basic"),
    
    # Derived metrics - balance sheet
    "Working Cash": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            min(get_metric("Cash", data, year) + get_metric("Marketable Securities", data, year), 0.02 * get_metric("Revenue", data, year))
        ),
    ),
    
    "Operating Current Assets": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            get_metric("Working Cash", data, year) + get_metric("Inventory", data, year) + get_metric("Accounts Receivable", data, year) + get_metric("Prepaid Assets", data, year)
        ),
    ),
    
    "Operating Current Liabilities": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            get_metric("Accounts Payable", data, year) + get_metric("Accrued Salaries", data, year) + get_metric("Deferred Revenue", data, year)
        ),
    ),
    
    "Net Working Capital": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            get_metric("Operating Current Assets", data, year) - get_metric("Operating Current Liabilities", data, year)
        ),
    ),
    
    "Invested Capital": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            get_metric("Net Working Capital", data, year) + get_metric("Property and Equipment", data, year) + get_metric("Intangible Assets", data, year) + get_metric("Other Assets", data, year)
        ),
    ),
    
    "Capital Turnover": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            get_metric("Revenue", data, year) / get_metric("Invested Capital", data, year)
        ),
    ),
    
    "Return on Invested Capital": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            get_metric("NOPAT", data, year) / get_metric("Invested Capital", data, year)
        ),
    ),
    
    "Quick Ratio": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            (get_metric("Cash", data, year) + get_metric("Marketable Securities", data, year) + get_metric("Accounts Receivable", data, year) + get_metric("Prepaid Assets", data, year)) / 
            (get_metric("Accounts Payable", data, year) + get_metric("Accrued Salaries", data, year) + get_metric("Deferred Revenue", data, year) + get_metric("Current Portion of Long-Term Debt", data, year))
        ),
    ),
    
    "Current Ratio": MetricDefinition(
        metric_type="derived",
        formula=lambda data, year: (
            (get_metric("Cash", data, year) + get_metric("Marketable Securities", data, year) + get_metric("Accounts Receivable", data, year) + get_metric("Prepaid Assets", data, year) + get_metric("Inventory", data, year)) / 
            (get_metric("Accounts Payable", data, year) + get_metric("Accrued Salaries", data, year) + get_metric("Deferred Revenue", data, year) + get_metric("Current Portion of Long-Term Debt", data, year))
        ),
    ),

}


def get_metric(metric_name: str, data: dict, year: Union[int, str], *args) -> float:
    """
    Computes the specified metric for the given year (and possibly additional years).
    - If 'metric_name' is a basic metric, returns data[year][metric_name].
    - If it's derived, recursively compute dependencies then run its formula.
    - Additional years can be passed in `args` if the formula needs them (e.g. for 'growth').
    """
    if metric_name not in METRIC_DEFINITIONS:
        return 0
    
    definition = METRIC_DEFINITIONS[metric_name]
    
    if definition.metric_type == "basic":
        # Just return the raw value from data
        return data[year][metric_name]
    
    elif definition.metric_type == "derived":
        # If the formula expects multiple years, pass them in
        # For a standard single-year derived metric, the formula only needs `data, year`.
        return definition.formula(data, year, *args)
    
def get_all_derived_metrics() -> List[str]:
    """
    Returns a list of all the names of derived metrics.
    """
    return [metric_name for metric_name, definition in METRIC_DEFINITIONS.items() if definition.metric_type == "derived"]

def get_all_basic_metrics() -> List[str]:
    return [metric_name for metric_name, definition in METRIC_DEFINITIONS.items() if definition.metric_type == "basic"]

def get_number_of_years_required(metric_name: str) -> int:
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
        },
        2022: {
            "Revenue": 2560,
            "Cost of Goods Sold": 846,
            "SG&A Expense": 405,
            "R&D Expense": 768,
            "Depreciation Expense": 283,
            "Interest Expense": 110,
            "Income Tax Expense": 209,
        }
    }

    # 1) Compute a simple derived metric for 2022
    net_margin_2022 = get_metric("Net Margin", sample_data, 2022)
    print(f"Net Margin (2022): {net_margin_2022:.2%}")
    
    net_margin_2021 = get_metric("Net Margin", sample_data, 2021)
    print(f"Net Margin (2021): {net_margin_2021:.2%}")
    
    gross_margin_2022 = get_metric("Gross Margin", sample_data, 2022)
    print(f"Gross Margin (2022): {gross_margin_2022:.2%}")
    
    gross_margin_2021 = get_metric("Gross Margin", sample_data, 2021)
    print(f"Gross Margin (2021): {gross_margin_2021:.2%}")
    
    operating_margin_2022 = get_metric("Operating Margin", sample_data, 2022)
    print(f"Operating Margin (2022): {operating_margin_2022:.2%}")
    
    operating_margin_2021 = get_metric("Operating Margin", sample_data, 2021)
    print(f"Operating Margin (2021): {operating_margin_2021:.2%}")
    
    net_margin_growth = get_metric("Absolute Change in Net Margin", sample_data, 2021, 2022)
    print(f"Growth in Net Margin (2021 to 2022): {net_margin_growth:.2%}")
