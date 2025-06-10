# backend/app/api/server_update.py
# Add these imports and route registrations to your existing server.py file

# Add these imports to the existing imports in server.py
from app.api.routes import (
    auth,
    queries,
    visualizations,
    analytics,
    database,
    admin,
    # Add these new route imports
    multi_tier,
    reports,
    settings,
    dashboards,
    suggestions,
    export,
    analytics_enhanced
)

# In the create_app function, after the existing route includes, add:

def create_app():
    # ... existing code ...
    
    # Existing routes
    app.include_router(auth.router)
    app.include_router(queries.router)
    app.include_router(visualizations.router)
    app.include_router(analytics.router)
    app.include_router(database.router)
    app.include_router(admin.router)
    
    # Add these new routes
    app.include_router(multi_tier.router)
    app.include_router(reports.router)
    app.include_router(settings.router)
    app.include_router(dashboards.router)
    app.include_router(suggestions.router)
    app.include_router(export.router)
    app.include_router(analytics_enhanced.router)
    
    # ... rest of existing code ...
    
    return app