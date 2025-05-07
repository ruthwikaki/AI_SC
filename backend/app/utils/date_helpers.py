"""
Date and time utilities.

This module provides functions for working with dates and time values.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import datetime, date, timedelta
import calendar
import pytz
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as dateutil_parse
from enum import Enum

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class TimeFrame(str, Enum):
    """Common time frames for reporting and analytics."""
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_QUARTER = "last_quarter"
    LAST_YEAR = "last_year"
    YEAR_TO_DATE = "year_to_date"
    QUARTER_TO_DATE = "quarter_to_date"
    MONTH_TO_DATE = "month_to_date"
    WEEK_TO_DATE = "week_to_date"
    CUSTOM = "custom"

def get_date_range(
    time_frame: Union[TimeFrame, str],
    custom_start_date: Optional[date] = None,
    custom_end_date: Optional[date] = None,
    timezone: str = "UTC"
) -> Tuple[date, date]:
    """
    Calculate the date range based on the time frame.
    
    Args:
        time_frame: Time frame to calculate
        custom_start_date: Custom start date (for TimeFrame.CUSTOM)
        custom_end_date: Custom end date (for TimeFrame.CUSTOM)
        timezone: Timezone for calculations
        
    Returns:
        Tuple of (start_date, end_date)
    """
    # Convert string to enum if needed
    if isinstance(time_frame, str):
        try:
            time_frame = TimeFrame(time_frame.lower())
        except ValueError:
            raise ValueError(f"Invalid time_frame: {time_frame}")
    
    # Get current date in the specified timezone
    tz = pytz.timezone(timezone)
    now = datetime.now(tz).date()
    
    # Handle custom time frame
    if time_frame == TimeFrame.CUSTOM:
        if not custom_start_date or not custom_end_date:
            raise ValueError("Custom time frame requires both start and end dates")
        return custom_start_date, custom_end_date
    
    # Calculate end date (usually today)
    end_date = now
    
    # Calculate start date based on time frame
    if time_frame == TimeFrame.LAST_DAY:
        start_date = now - timedelta(days=1)
    
    elif time_frame == TimeFrame.LAST_WEEK:
        start_date = now - timedelta(days=7)
    
    elif time_frame == TimeFrame.LAST_MONTH:
        start_date = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
    
    elif time_frame == TimeFrame.LAST_QUARTER:
        current_quarter_month = ((now.month - 1) // 3) * 3 + 1
        current_quarter_start = now.replace(month=current_quarter_month, day=1)
        last_quarter_end = current_quarter_start - timedelta(days=1)
        last_quarter_month = ((last_quarter_end.month - 1) // 3) * 3 + 1
        start_date = last_quarter_end.replace(month=last_quarter_month, day=1)
    
    elif time_frame == TimeFrame.LAST_YEAR:
        start_date = now.replace(year=now.year-1, month=now.month, day=now.day)
    
    elif time_frame == TimeFrame.YEAR_TO_DATE:
        start_date = now.replace(month=1, day=1)
    
    elif time_frame == TimeFrame.QUARTER_TO_DATE:
        current_quarter_month = ((now.month - 1) // 3) * 3 + 1
        start_date = now.replace(month=current_quarter_month, day=1)
    
    elif time_frame == TimeFrame.MONTH_TO_DATE:
        start_date = now.replace(day=1)
    
    elif time_frame == TimeFrame.WEEK_TO_DATE:
        # Assuming weeks start on Monday (weekday=0)
        start_date = now - timedelta(days=now.weekday())
    
    else:
        raise ValueError(f"Unsupported time frame: {time_frame}")
    
    return start_date, end_date

def parse_date(date_str: str, default: Optional[date] = None) -> Optional[date]:
    """
    Parse a date string into a date object.
    
    Args:
        date_str: Date string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed date or default
    """
    if not date_str:
        return default
    
    try:
        # Try to parse the date using dateutil
        dt = dateutil_parse(date_str)
        return dt.date()
    except Exception as e:
        logger.debug(f"Error parsing date '{date_str}': {str(e)}")
        return default

def format_date(d: date, format_str: str = "%Y-%m-%d") -> str:
    """
    Format a date as a string.
    
    Args:
        d: Date to format
        format_str: Format string
        
    Returns:
        Formatted date string
    """
    if not d:
        return ""
    
    return d.strftime(format_str)

def get_start_of_month(d: date) -> date:
    """
    Get the first day of the month for a date.
    
    Args:
        d: Date
        
    Returns:
        First day of the month
    """
    return d.replace(day=1)

def get_end_of_month(d: date) -> date:
    """
    Get the last day of the month for a date.
    
    Args:
        d: Date
        
    Returns:
        Last day of the month
    """
    # Find the first day of the next month and subtract 1 day
    next_month = d.replace(day=28) + timedelta(days=4)  # Move to next month
    return next_month.replace(day=1) - timedelta(days=1)

def get_start_of_quarter(d: date) -> date:
    """
    Get the first day of the quarter for a date.
    
    Args:
        d: Date
        
    Returns:
        First day of the quarter
    """
    quarter_month = ((d.month - 1) // 3) * 3 + 1
    return d.replace(month=quarter_month, day=1)

def get_end_of_quarter(d: date) -> date:
    """
    Get the last day of the quarter for a date.
    
    Args:
        d: Date
        
    Returns:
        Last day of the quarter
    """
    quarter_month = ((d.month - 1) // 3) * 3 + 3
    return get_end_of_month(d.replace(month=quarter_month))

def get_start_of_year(d: date) -> date:
    """
    Get the first day of the year for a date.
    
    Args:
        d: Date
        
    Returns:
        First day of the year
    """
    return d.replace(month=1, day=1)

def get_end_of_year(d: date) -> date:
    """
    Get the last day of the year for a date.
    
    Args:
        d: Date
        
    Returns:
        Last day of the year
    """
    return d.replace(month=12, day=31)

def date_range_inclusive(start_date: date, end_date: date) -> List[date]:
    """
    Generate a list of dates in a range (inclusive).
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of dates
    """
    delta = (end_date - start_date).days + 1
    return [start_date + timedelta(days=i) for i in range(delta)]

def month_range(start_date: date, end_date: date) -> List[Tuple[date, date]]:
    """
    Generate a list of (month_start, month_end) tuples for a date range.
    
    Args:
        start_date: Start date
        end_date: End date
        
    Returns:
        List of month ranges
    """
    months = []
    current = get_start_of_month(start_date)
    
    while current <= end_date:
        month_end = min(get_end_of_month(current), end_date)
        months.append((current, month_end))
        
        # Move to next month
        current = (current + relativedelta(months=1)).replace(day=1)
    
    return months

def get_fiscal_year_dates(
    date_obj: date,
    fiscal_year_start_month: int = 1
) -> Tuple[date, date]:
    """
    Get the start and end dates for the fiscal year containing a date.
    
    Args:
        date_obj: Date to find fiscal year for
        fiscal_year_start_month: Month when fiscal year starts (1-12)
        
    Returns:
        Tuple of (fiscal_year_start, fiscal_year_end)
    """
    if not 1 <= fiscal_year_start_month <= 12:
        raise ValueError("fiscal_year_start_month must be between 1 and 12")
    
    # Determine if the date is in the current or previous fiscal year
    if date_obj.month >= fiscal_year_start_month:
        # Date is in the current fiscal year
        fiscal_start_year = date_obj.year
    else:
        # Date is in the previous fiscal year
        fiscal_start_year = date_obj.year - 1
    
    # Calculate start date
    fiscal_start = date(fiscal_start_year, fiscal_year_start_month, 1)
    
    # Calculate end date (last day of the month before the start month of next fiscal year)
    if fiscal_year_start_month == 1:
        fiscal_end = date(fiscal_start_year + 1, 12, 31)
    else:
        fiscal_end_year = fiscal_start_year + 1
        fiscal_end_month = fiscal_year_start_month - 1
        last_day = calendar.monthrange(fiscal_end_year, fiscal_end_month)[1]
        fiscal_end = date(fiscal_end_year, fiscal_end_month, last_day)
    
    return fiscal_start, fiscal_end

def get_business_days(start_date: date, end_date: date, holidays: List[date] = None) -> int:
    """
    Calculate the number of business days between two dates.
    
    Args:
        start_date: Start date
        end_date: End date
        holidays: Optional list of holiday dates to exclude
        
    Returns:
        Number of business days
    """
    if not holidays:
        holidays = []
    
    # Convert holidays to a set for faster lookup
    holiday_set = set(holidays)
    
    # Initialize counter
    business_days = 0
    current_date = start_date
    
    # Count business days
    while current_date <= end_date:
        # Check if it's a weekday (Monday=0, Sunday=6)
        if current_date.weekday() < 5 and current_date not in holiday_set:
            business_days += 1
        
        current_date += timedelta(days=1)
    
    return business_days

def get_next_business_day(
    from_date: date,
    holidays: List[date] = None,
    skip_days: int = 1
) -> date:
    """
    Get the next business day after a date.
    
    Args:
        from_date: Starting date
        holidays: Optional list of holiday dates to exclude
        skip_days: Number of business days to skip
        
    Returns:
        Next business day
    """
    if not holidays:
        holidays = []
    
    # Convert holidays to a set for faster lookup
    holiday_set = set(holidays)
    
    # Initialize counter
    days_skipped = 0
    current_date = from_date + timedelta(days=1)  # Start from next day
    
    # Find next business day
    while days_skipped < skip_days:
        # Check if it's a business day
        if current_date.weekday() < 5 and current_date not in holiday_set:
            days_skipped += 1
            
            # If we've skipped enough days, return the current date
            if days_skipped >= skip_days:
                return current_date
        
        current_date += timedelta(days=1)
    
    return current_date

def age_in_days(d: date, reference_date: Optional[date] = None) -> int:
    """
    Calculate the age of a date in days.
    
    Args:
        d: Date to calculate age for
        reference_date: Reference date (default: today)
        
    Returns:
        Age in days
    """
    if not reference_date:
        reference_date = date.today()
    
    return (reference_date - d).days

def group_dates_by_period(
    dates: List[date],
    period: str = "month"
) -> Dict[str, List[date]]:
    """
    Group dates by a time period.
    
    Args:
        dates: List of dates to group
        period: Grouping period ("day", "week", "month", "quarter", "year")
        
    Returns:
        Dictionary of grouped dates
    """
    grouped = {}
    
    for d in dates:
        if period == "day":
            key = d.strftime("%Y-%m-%d")
        elif period == "week":
            # ISO week format (year-W##)
            key = d.strftime("%Y-W%U")
        elif period == "month":
            key = d.strftime("%Y-%m")
        elif period == "quarter":
            quarter = (d.month - 1) // 3 + 1
            key = f"{d.year}-Q{quarter}"
        elif period == "year":
            key = str(d.year)
        else:
            raise ValueError(f"Unsupported period: {period}")
        
        if key not in grouped:
            grouped[key] = []
        
        grouped[key].append(d)
    
    return grouped

def is_weekend(d: date) -> bool:
    """
    Check if a date is a weekend (Saturday or Sunday).
    
    Args:
        d: Date to check
        
    Returns:
        True if weekend, False otherwise
    """
    return d.weekday() >= 5  # 5=Saturday, 6=Sunday

def is_same_day(date1: date, date2: date) -> bool:
    """
    Check if two dates represent the same day.
    
    Args:
        date1: First date
        date2: Second date
        
    Returns:
        True if same day, False otherwise
    """
    return (date1.year == date2.year and 
            date1.month == date2.month and 
            date1.day == date2.day)

def get_date_periods(
    start_date: date,
    end_date: date,
    period: str = "month"
) -> List[Tuple[date, date]]:
    """
    Generate a list of period ranges within a date range.
    
    Args:
        start_date: Start date
        end_date: End date
        period: Period type ("day", "week", "month", "quarter", "year")
        
    Returns:
        List of (period_start, period_end) tuples
    """
    if period == "day":
        return [(d, d) for d in date_range_inclusive(start_date, end_date)]
    
    elif period == "week":
        result = []
        current = start_date - timedelta(days=start_date.weekday())  # Start of week
        if current < start_date:
            current += timedelta(days=7)
        
        while current <= end_date:
            week_end = current + timedelta(days=6)
            if week_end > end_date:
                week_end = end_date
            
            result.append((current, week_end))
            current += timedelta(days=7)
        
        return result
    
    elif period == "month":
        return month_range(start_date, end_date)
    
    elif period == "quarter":
        result = []
        current = get_start_of_quarter(start_date)
        
        while current <= end_date:
            quarter_end = get_end_of_quarter(current)
            if quarter_end > end_date:
                quarter_end = end_date
            
            result.append((current, quarter_end))
            current = quarter_end + timedelta(days=1)
            current = get_start_of_quarter(current)
        
        return result
    
    elif period == "year":
        result = []
        current = get_start_of_year(start_date)
        
        while current <= end_date:
            year_end = get_end_of_year(current)
            if year_end > end_date:
                year_end = end_date
            
            result.append((current, year_end))
            current = year_end + timedelta(days=1)
            current = get_start_of_year(current)
        
        return result
    
    else:
        raise ValueError(f"Unsupported period: {period}")

def date_diff_in_words(
    from_date: date,
    to_date: Optional[date] = None,
    detailed: bool = False
) -> str:
    """
    Get a human-readable string describing the difference between two dates.
    
    Args:
        from_date: Starting date
        to_date: Ending date (default: today)
        detailed: Whether to include more detail
        
    Returns:
        Human-readable date difference
    """
    if not to_date:
        to_date = date.today()
    
    # Calculate the difference
    delta = relativedelta(to_date, from_date)
    
    # Simple case
    if not detailed:
        if delta.years > 0:
            return f"{delta.years} year{'s' if delta.years != 1 else ''} ago"
        if delta.months > 0:
            return f"{delta.months} month{'s' if delta.months != 1 else ''} ago"
        if delta.days > 7:
            weeks = delta.days // 7
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        if delta.days > 0:
            return f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
        return "today"
    
    # Detailed case
    parts = []
    if delta.years > 0:
        parts.append(f"{delta.years} year{'s' if delta.years != 1 else ''}")
    if delta.months > 0:
        parts.append(f"{delta.months} month{'s' if delta.months != 1 else ''}")
    if delta.days > 0:
        parts.append(f"{delta.days} day{'s' if delta.days != 1 else ''}")
    
    if parts:
        return ", ".join(parts) + " ago"
    return "today"

def get_quarter_name(d: date) -> str:
    """
    Get the quarter name for a date (e.g., "Q1 2023").
    
    Args:
        d: Date
        
    Returns:
        Quarter name
    """
    quarter = (d.month - 1) // 3 + 1
    return f"Q{quarter} {d.year}"

def get_month_name(d: date, short: bool = False) -> str:
    """
    Get the month name for a date.
    
    Args:
        d: Date
        short: Whether to use short month name
        
    Returns:
        Month name
    """
    if short:
        return d.strftime("%b")
    return d.strftime("%B")

def days_until(target_date: date, from_date: Optional[date] = None) -> int:
    """
    Calculate the number of days until a target date.
    
    Args:
        target_date: Target date
        from_date: Starting date (default: today)
        
    Returns:
        Number of days
    """
    if not from_date:
        from_date = date.today()
    
    return (target_date - from_date).days

def is_future_date(d: date, reference_date: Optional[date] = None) -> bool:
    """
    Check if a date is in the future.
    
    Args:
        d: Date to check
        reference_date: Reference date (default: today)
        
    Returns:
        True if future date, False otherwise
    """
    if not reference_date:
        reference_date = date.today()
    
    return d > reference_date

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a datetime as a string.
    
    Args:
        dt: Datetime to format
        format_str: Format string
        
    Returns:
        Formatted datetime string
    """
    if not dt:
        return ""
    
    return dt.strftime(format_str)

def get_utc_now() -> datetime:
    """
    Get the current UTC datetime.
    
    Returns:
        Current UTC datetime
    """
    return datetime.now(pytz.UTC)

def localize_datetime(dt: datetime, timezone: str = "UTC") -> datetime:
    """
    Localize a datetime to a specific timezone.
    
    Args:
        dt: Datetime to localize
        timezone: Target timezone
        
    Returns:
        Localized datetime
    """
    if dt.tzinfo is None:
        # Assume UTC if no timezone is specified
        dt = pytz.UTC.localize(dt)
    
    target_tz = pytz.timezone(timezone)
    return dt.astimezone(target_tz)

def parse_datetime(datetime_str: str, default: Optional[datetime] = None) -> Optional[datetime]:
    """
    Parse a datetime string into a datetime object.
    
    Args:
        datetime_str: Datetime string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed datetime or default
    """
    if not datetime_str:
        return default
    
    try:
        # Try to parse the datetime using dateutil
        return dateutil_parse(datetime_str)
    except Exception as e:
        logger.debug(f"Error parsing datetime '{datetime_str}': {str(e)}")
        return default