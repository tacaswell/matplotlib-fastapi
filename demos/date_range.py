"""
Date range visualization demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/date_range.py
"""

import numpy as np
from datetime import date, timedelta
from typing import Literal
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig


class DateRangeParams(BaseModel):
    """Parameters for date range visualization - demonstrates date types."""

    start_date: date = Field(
        default=date(2024, 1, 1),
        description="Start date for the time series",
    )
    end_date: date = Field(
        default=date(2024, 12, 31),
        description="End date for the time series",
    )
    sampling: Literal["daily", "weekly", "monthly"] = Field(
        default="daily",
        description="Data sampling frequency",
    )
    show_weekends: bool = Field(
        default=False,
        description="Highlight weekends in gray",
    )


def create_date_range_plot(fig: Figure, params: DateRangeParams) -> None:
    """Render a date-based plot demonstrating date pickers.

    This plot demonstrates:
    - date type (HTML5 date picker)
    - Date range validation and formatting
    - Date-based calculations
    """
    import matplotlib.dates as mdates

    ax = fig.subplots()

    # Calculate number of days
    num_days = (params.end_date - params.start_date).days + 1
    if num_days <= 0:
        # Handle invalid date range
        ax.text(
            0.5,
            0.5,
            "Invalid date range!\nStart date must be before end date.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="red",
        )
        return

    # Generate dates based on sampling
    if params.sampling == "daily":
        dates = [params.start_date + timedelta(days=i) for i in range(num_days)]
        num_points = num_days
    elif params.sampling == "weekly":
        num_points = num_days // 7 + 1
        dates = [params.start_date + timedelta(weeks=i) for i in range(num_points)]
    else:  # monthly
        dates = []
        current = params.start_date
        while current <= params.end_date:
            dates.append(current)
            # Move to next month
            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)
        num_points = len(dates)

    # Generate synthetic data (random walk)
    np.random.seed(hash(str(params.start_date)) % 2**32)
    values = np.cumsum(np.random.randn(num_points)) + 100

    # Plot the main line
    ax.plot(dates, values, "o-", linewidth=2, markersize=4, label="Value")

    # Highlight weekends if requested
    if params.show_weekends and params.sampling == "daily":
        for i, d in enumerate(dates):
            if d.weekday() >= 5:  # Saturday=5, Sunday=6
                ax.axvspan(d, d + timedelta(days=1), color="gray", alpha=0.2)

    # Format x-axis
    if num_days <= 60:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    elif num_days <= 365:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()  # Rotate date labels

    # Add grid and labels
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(
        f"Date Range: {params.start_date} to {params.end_date} ({params.sampling})"
    )
    ax.legend()


plots = {
    "date_range": PlotConfig(
        description="Date range plot with HTML5 date pickers",
        init=InitConfig(function=create_date_range_plot, params_model=DateRangeParams),
    ),
}
