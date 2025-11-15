class GoogleCalendarBaseTool:
    """Base class for Google Calendar tools using a service account."""
    def __init__(self, api_resource):
        self.api_resource = api_resource

    @classmethod
    def from_api_resource(cls, api_resource):
        return cls(api_resource=api_resource)
    
