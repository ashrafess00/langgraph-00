from dotenv import load_dotenv
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from langchain_core.tools import tool
from datetime import datetime
from typing import Optional

# Try to import pytz, fallback to simpler timezone handling
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

load_dotenv()

# Calendar configuration
SERVICE_ACCOUNT_FILE = "calendar-service-account.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Get calendar ID from env or use default
google_calendar_id = os.getenv("GOOGLE_CALENDAR_ID")
DEFAULT_CALENDAR_ID = "69c975bfdf793293bed86e01be053cbdb1d5ddd560886ea390aab898ed608d93@group.calendar.google.com"
CALENDAR_ID = google_calendar_id if google_calendar_id else DEFAULT_CALENDAR_ID

print(f"📅 Using Calendar ID: {CALENDAR_ID}")

# Setup Google Calendar API service
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
calendar_service = build('calendar', 'v3', credentials=credentials)

print("✅ Google Calendar service initialized")


@tool
def create_calendar_event(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    timezone: str = "UTC",
    location: Optional[str] = None,
    description: Optional[str] = None,
    reminder_minutes: Optional[int] = None
) -> str:
    """
    Create an event on Google Calendar.
    
    Args:
        summary: Title of the event (required)
        start_datetime: Start date and time in format 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS' (required)
        end_datetime: End date and time in format 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DDTHH:MM:SS' (required)
        timezone: Timezone (e.g., 'America/New_York', 'Asia/Kolkata', 'UTC'). Defaults to 'UTC'
        location: Location of the event (optional)
        description: Description of the event (optional)
        reminder_minutes: Minutes before event to send reminder (optional, e.g., 60 for 1 hour before)
    
    Returns:
        A message indicating success or failure with event details
    """
    try:
        # Parse datetime strings - convert to ISO format
        # Handle both 'YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS' formats
        start_str = start_datetime.replace(' ', 'T') if 'T' not in start_datetime else start_datetime
        end_str = end_datetime.replace(' ', 'T') if 'T' not in end_datetime else end_datetime
        
        # If no timezone info, assume it's in the specified timezone
        if not start_str.endswith('Z') and '+' not in start_str[-6:] and '-' not in start_str[-6:]:
            # Parse datetime
            try:
                start_dt = datetime.fromisoformat(start_str)
                end_dt = datetime.fromisoformat(end_str)
            except ValueError:
                # Try alternative format
                start_dt = datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%S')
                end_dt = datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%S')
            
            # Make timezone-aware if pytz is available
            if HAS_PYTZ:
                tz = pytz.timezone(timezone)
                start_dt = tz.localize(start_dt)
                end_dt = tz.localize(end_dt)
                start_rfc = start_dt.isoformat()
                end_rfc = end_dt.isoformat()
            else:
                # Simple format: just use the datetime with timezone name
                # Google Calendar API will interpret this
                start_rfc = f"{start_str}+00:00"  # Assume UTC offset, API will use timeZone param
                end_rfc = f"{end_str}+00:00"
        else:
            start_rfc = start_str
            end_rfc = end_str
        
        # Build event body
        event_body = {
            'summary': summary,
            'start': {
                'dateTime': start_rfc,
                'timeZone': timezone,
            },
            'end': {
                'dateTime': end_rfc,
                'timeZone': timezone,
            },
        }
        
        # Add optional fields
        if location:
            event_body['location'] = location
        if description:
            event_body['description'] = description
        if reminder_minutes is not None:
            event_body['reminders'] = {
                'useDefault': False,
                'overrides': [{"method": "popup", "minutes": reminder_minutes}]
            }
        
        # Create the event
        event = calendar_service.events().insert(
            calendarId=CALENDAR_ID,
            body=event_body
        ).execute()
        
        # Return success message with event details
        event_link = event.get('htmlLink', 'N/A')
        event_id = event.get('id', 'N/A')
        
        return f"✅ Event created successfully!\n" \
               f"Title: {summary}\n" \
               f"Start: {start_rfc}\n" \
               f"End: {end_rfc}\n" \
               f"Calendar: {CALENDAR_ID}\n" \
               f"Event ID: {event_id}\n" \
               f"Link: {event_link}"
               
    except Exception as e:
        return f"❌ Error creating event: {str(e)}"


# Create the tools list
tools = [create_calendar_event]

# Export for use in other files
__all__ = ['tools', 'calendar_service', 'CALENDAR_ID']

print(f"✅ Created {len(tools)} custom tool(s)")
