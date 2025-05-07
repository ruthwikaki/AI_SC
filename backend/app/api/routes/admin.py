from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime, date, timedelta
import uuid

from app.db.interfaces.user_interface import User
from app.security.rbac_manager import check_permission, get_role_permissions, update_role_permissions
from app.security.audit_logger import get_audit_logs
from app.llm.controller.active_model_manager import get_active_model, set_active_model, get_available_models
from app.utils.logger import get_logger
from app.db.schema.schema_mapper import get_domain_mappings, update_domain_mappings
from app.db.mirroring.change_monitor import get_sync_status, trigger_sync

from app.api.routes.auth import get_current_active_user

# Initialize logger
logger = get_logger(__name__)

# Router
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_active_user)],
    responses={401: {"description": "Unauthorized"}}
)

# Models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str
    client_id: Optional[str] = None
    is_active: bool = True

class UserUpdate(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None
    client_id: Optional[str] = None

class Client(BaseModel):
    id: str
    name: str
    status: str
    created_at: datetime
    primary_contact: str
    trial_ends_at: Optional[datetime] = None
    subscription_plan: Optional[str] = None
    subscription_status: Optional[str] = None
    data_size_mb: Optional[int] = None
    last_sync: Optional[datetime] = None
    domain: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

class ClientCreate(BaseModel):
    name: str
    primary_contact: str
    domain: Optional[str] = None
    subscription_plan: Optional[str] = None
    trial_period_days: int = 30

class ClientUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    primary_contact: Optional[str] = None
    subscription_plan: Optional[str] = None
    subscription_status: Optional[str] = None
    domain: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

class Role(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    permissions: List[str]
    is_system_role: bool = False
    created_at: datetime

class RoleCreate(BaseModel):
    name: str
    description: Optional[str] = None
    permissions: List[str]

class RoleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    permissions: Optional[List[str]] = None

class AuditLog(BaseModel):
    id: str
    timestamp: datetime
    user_id: str
    username: str
    action: str
    resource: str
    resource_id: Optional[str] = None
    client_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class SyncStatus(BaseModel):
    client_id: str
    connection_id: str
    last_sync: Optional[datetime] = None
    status: str
    tables_synced: int
    total_tables: int
    rows_synced: int
    total_rows: int
    error_message: Optional[str] = None
    is_initial_sync: bool
    estimated_completion: Optional[datetime] = None

class DomainMapping(BaseModel):
    client_id: str
    custom_table: str
    custom_column: str
    domain_concept: str
    confidence: float
    manual_override: bool = False
    last_updated: datetime

# Routes
@router.get("/users", response_model=List[User])
async def get_users(
    client_id: Optional[str] = None,
    role: Optional[str] = None,
    is_active: Optional[bool] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Get all users with optional filtering"""
    # Check user has permission
    check_permission(current_user.role, "admin:users:view")
    
    # For SaaS admins, client_id can be specified
    # For client admins, only their own client_id is valid
    if current_user.role != "admin" and client_id != current_user.client_id:
        client_id = current_user.client_id
    
    try:
        # Get users from database
        from app.db.interfaces.user_interface import UserInterface
        user_interface = UserInterface()
        users = await user_interface.get_users(
            client_id=client_id,
            role=role,
            is_active=is_active
        )
        
        logger.info(f"Retrieved users, count: {len(users)}")
        return users
        
    except Exception as e:
        logger.error(f"Error retrieving users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving users: {str(e)}"
        )

@router.post("/users", response_model=User)
async def create_user(
    user_create: UserCreate,
    current_user: User = Depends(get_current_active_user),
):
    """Create a new user"""
    # Check user has permission
    check_permission(current_user.role, "admin:users:create")
    
    # For client admins, only their own client_id is valid
    if current_user.role != "admin" and user_create.client_id != current_user.client_id:
        user_create.client_id = current_user.client_id
    
    # Check if client_id is provided
    if not user_create.client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Create user in database
        from app.db.interfaces.user_interface import UserInterface
        user_interface = UserInterface()
        
        # Check if username already exists
        existing_user = await user_interface.get_user_by_username(user_create.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )
        
        # Check if email already exists
        existing_email = await user_interface.get_user_by_email(user_create.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already exists"
            )
        
        # Hash password
        from app.security.encryption import get_password_hash
        hashed_password = get_password_hash(user_create.password)
        
        # Create user
        new_user = await user_interface.create_user(
            username=user_create.username,
            email=user_create.email,
            hashed_password=hashed_password,
            role=user_create.role,
            client_id=user_create.client_id,
            is_active=user_create.is_active
        )
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="create",
            resource="user",
            resource_id=new_user.id,
            client_id=new_user.client_id,
            details={"username": new_user.username, "role": new_user.role}
        )
        
        logger.info(f"Created user: {user_create.username}")
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@router.get("/users/{user_id}", response_model=User)
async def get_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific user by ID"""
    # Check user has permission
    check_permission(current_user.role, "admin:users:view")
    
    try:
        # Get user from database
        from app.db.interfaces.user_interface import UserInterface
        user_interface = UserInterface()
        user = await user_interface.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # For client admins, can only view users in their client
        if current_user.role != "admin" and user.client_id != current_user.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this user"
            )
        
        logger.info(f"Retrieved user: {user.username}")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user: {str(e)}"
        )

@router.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """Update a user"""
    # Check user has permission
    check_permission(current_user.role, "admin:users:update")
    
    try:
        # Get user from database
        from app.db.interfaces.user_interface import UserInterface
        user_interface = UserInterface()
        user = await user_interface.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # For client admins, can only update users in their client
        if current_user.role != "admin" and user.client_id != current_user.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this user"
            )
        
        # Collect update data
        update_data = {}
        if user_update.email is not None:
            update_data["email"] = user_update.email
        if user_update.role is not None:
            update_data["role"] = user_update.role
        if user_update.is_active is not None:
            update_data["is_active"] = user_update.is_active
        if user_update.client_id is not None and current_user.role == "admin":
            update_data["client_id"] = user_update.client_id
        
        # Update user
        updated_user = await user_interface.update_user(user_id, update_data)
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="update",
            resource="user",
            resource_id=user_id,
            client_id=user.client_id,
            details=update_data
        )
        
        logger.info(f"Updated user: {user.username}")
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating user: {str(e)}"
        )

@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Delete a user"""
    # Check user has permission
    check_permission(current_user.role, "admin:users:delete")
    
    try:
        # Get user from database
        from app.db.interfaces.user_interface import UserInterface
        user_interface = UserInterface()
        user = await user_interface.get_user(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # For client admins, can only delete users in their client
        if current_user.role != "admin" and user.client_id != current_user.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this user"
            )
        
        # Cannot delete self
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Delete user
        await user_interface.delete_user(user_id)
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="delete",
            resource="user",
            resource_id=user_id,
            client_id=user.client_id,
            details={"username": user.username}
        )
        
        logger.info(f"Deleted user: {user.username}")
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting user: {str(e)}"
        )

@router.get("/clients", response_model=List[Client])
async def get_clients(
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Get all clients with optional filtering"""
    # Check user has permission
    check_permission(current_user.role, "admin:clients:view")
    
    # Only system admins can view all clients
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view all clients"
        )
    
    try:
        # Get clients from database
        # In a real implementation, you would fetch from database
        # For demonstration, we'll return mock data
        
        clients = [
            Client(
                id="client-1",
                name="Acme Corporation",
                status="active",
                created_at=datetime.now() - timedelta(days=30),
                primary_contact="john.doe@acme.com",
                trial_ends_at=datetime.now() + timedelta(days=60),
                subscription_plan="enterprise",
                subscription_status="active",
                data_size_mb=1250,
                last_sync=datetime.now() - timedelta(hours=2),
                domain="manufacturing"
            ),
            Client(
                id="client-2",
                name="TechStart Inc",
                status="active",
                created_at=datetime.now() - timedelta(days=15),
                primary_contact="jane.smith@techstart.com",
                trial_ends_at=datetime.now() + timedelta(days=15),
                subscription_plan="pro",
                subscription_status="trial",
                data_size_mb=750,
                last_sync=datetime.now() - timedelta(hours=6),
                domain="electronics"
            )
        ]
        
        # Apply status filter if provided
        if status:
            clients = [c for c in clients if c.status == status]
        
        logger.info(f"Retrieved clients, count: {len(clients)}")
        return clients
        
    except Exception as e:
        logger.error(f"Error retrieving clients: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving clients: {str(e)}"
        )

@router.post("/clients", response_model=Client)
async def create_client(
    client_create: ClientCreate,
    current_user: User = Depends(get_current_active_user),
):
    """Create a new client"""
    # Check user has permission
    check_permission(current_user.role, "admin:clients:create")
    
    # Only system admins can create clients
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create clients"
        )
    
    try:
        # In a real implementation, you would save to database
        # For demonstration, we'll create a mock client
        
        # Calculate trial end date
        trial_ends_at = datetime.now() + timedelta(days=client_create.trial_period_days)
        
        new_client = Client(
            id=f"client-{uuid.uuid4().hex[:8]}",
            name=client_create.name,
            status="active",
            created_at=datetime.now(),
            primary_contact=client_create.primary_contact,
            trial_ends_at=trial_ends_at,
            subscription_plan=client_create.subscription_plan or "trial",
            subscription_status="trial",
            domain=client_create.domain,
            data_size_mb=0,
            last_sync=None
        )
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="create",
            resource="client",
            resource_id=new_client.id,
            details={"name": new_client.name}
        )
        
        logger.info(f"Created client: {client_create.name}")
        return new_client
        
    except Exception as e:
        logger.error(f"Error creating client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating client: {str(e)}"
        )

@router.get("/clients/{client_id}", response_model=Client)
async def get_client(
    client_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific client by ID"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:clients:view")
    else:
        # Client admins can only view their own client
        check_permission(current_user.role, "admin:client:view")
        if client_id != current_user.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this client"
            )
    
    try:
        # In a real implementation, you would fetch from database
        # For demonstration, we'll return mock data
        
        # Mock client data
        if client_id == "client-1":
            client = Client(
                id="client-1",
                name="Acme Corporation",
                status="active",
                created_at=datetime.now() - timedelta(days=30),
                primary_contact="john.doe@acme.com",
                trial_ends_at=datetime.now() + timedelta(days=60),
                subscription_plan="enterprise",
                subscription_status="active",
                data_size_mb=1250,
                last_sync=datetime.now() - timedelta(hours=2),
                domain="manufacturing"
            )
        elif client_id == "client-2":
            client = Client(
                id="client-2",
                name="TechStart Inc",
                status="active",
                created_at=datetime.now() - timedelta(days=15),
                primary_contact="jane.smith@techstart.com",
                trial_ends_at=datetime.now() + timedelta(days=15),
                subscription_plan="pro",
                subscription_status="trial",
                data_size_mb=750,
                last_sync=datetime.now() - timedelta(hours=6),
                domain="electronics"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found"
            )
        
        logger.info(f"Retrieved client: {client.name}")
        return client
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving client: {str(e)}"
        )

@router.put("/clients/{client_id}", response_model=Client)
async def update_client(
    client_id: str,
    client_update: ClientUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """Update a client"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:clients:update")
    else:
        # Client admins can only update their own client and with limited fields
        check_permission(current_user.role, "admin:client:update")
        if client_id != current_user.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this client"
            )
        # Limited fields for client admins
        allowed_fields = ["name", "primary_contact", "settings"]
        for field in client_update.dict(exclude_unset=True).keys():
            if field not in allowed_fields:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not authorized to update field: {field}"
                )
    
    try:
        # In a real implementation, you would update in database
        # For demonstration, we'll update mock data
        
        # Mock client data
        if client_id == "client-1":
            client = Client(
                id="client-1",
                name="Acme Corporation",
                status="active",
                created_at=datetime.now() - timedelta(days=30),
                primary_contact="john.doe@acme.com",
                trial_ends_at=datetime.now() + timedelta(days=60),
                subscription_plan="enterprise",
                subscription_status="active",
                data_size_mb=1250,
                last_sync=datetime.now() - timedelta(hours=2),
                domain="manufacturing"
            )
        elif client_id == "client-2":
            client = Client(
                id="client-2",
                name="TechStart Inc",
                status="active",
                created_at=datetime.now() - timedelta(days=15),
                primary_contact="jane.smith@techstart.com",
                trial_ends_at=datetime.now() + timedelta(days=15),
                subscription_plan="pro",
                subscription_status="trial",
                data_size_mb=750,
                last_sync=datetime.now() - timedelta(hours=6),
                domain="electronics"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Client not found"
            )
        
        # Apply updates
        update_data = {k: v for k, v in client_update.dict(exclude_unset=True).items()}
        for key, value in update_data.items():
            setattr(client, key, value)
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="update",
            resource="client",
            resource_id=client_id,
            details=update_data
        )
        
        logger.info(f"Updated client: {client.name}")
        return client
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating client: {str(e)}"
        )

@router.get("/roles", response_model=List[Role])
async def get_roles(
    current_user: User = Depends(get_current_active_user),
):
    """Get all roles"""
    # Check user has permission
    check_permission(current_user.role, "admin:roles:view")
    
    try:
        # In a real implementation, you would fetch from database
        # For demonstration, we'll return mock data
        
        roles = [
            Role(
                id="role-1",
                name="admin",
                description="System administrator with full access",
                permissions=["admin:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            ),
            Role(
                id="role-2",
                name="client_admin",
                description="Client administrator with limited admin access",
                permissions=["admin:client:*", "admin:users:*", "queries:*", "visualizations:*", "analytics:*", "database:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            ),
            Role(
                id="role-3",
                name="analyst",
                description="Data analyst with query and visualization access",
                permissions=["queries:*", "visualizations:*", "analytics:*", "database:read"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            ),
            Role(
                id="role-4",
                name="viewer",
                description="Read-only access to dashboards and visualizations",
                permissions=["queries:execute", "visualizations:view"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        ]
        
        logger.info(f"Retrieved roles, count: {len(roles)}")
        return roles
        
    except Exception as e:
        logger.error(f"Error retrieving roles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving roles: {str(e)}"
        )

@router.post("/roles", response_model=Role)
async def create_role(
    role_create: RoleCreate,
    current_user: User = Depends(get_current_active_user),
):
    """Create a new role"""
    # Check user has permission
    check_permission(current_user.role, "admin:roles:create")
    
    # Only system admins can create roles
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create roles"
        )
    
    try:
        # In a real implementation, you would save to database
        # For demonstration, we'll create a mock role
        
        new_role = Role(
            id=f"role-{uuid.uuid4().hex[:8]}",
            name=role_create.name,
            description=role_create.description,
            permissions=role_create.permissions,
            is_system_role=False,
            created_at=datetime.now()
        )
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="create",
            resource="role",
            resource_id=new_role.id,
            details={"name": new_role.name, "permissions": new_role.permissions}
        )
        
        logger.info(f"Created role: {role_create.name}")
        return new_role
        
    except Exception as e:
        logger.error(f"Error creating role: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating role: {str(e)}"
        )

@router.get("/roles/{role_id}", response_model=Role)
async def get_role(
    role_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Get a specific role by ID"""
    # Check user has permission
    check_permission(current_user.role, "admin:roles:view")
    
    try:
        # In a real implementation, you would fetch from database
        # For demonstration, we'll return mock data
        
        # Mock role data based on ID
        if role_id == "role-1":
            role = Role(
                id="role-1",
                name="admin",
                description="System administrator with full access",
                permissions=["admin:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        elif role_id == "role-2":
            role = Role(
                id="role-2",
                name="client_admin",
                description="Client administrator with limited admin access",
                permissions=["admin:client:*", "admin:users:*", "queries:*", "visualizations:*", "analytics:*", "database:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        logger.info(f"Retrieved role: {role.name}")
        return role
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving role: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving role: {str(e)}"
        )

@router.put("/roles/{role_id}", response_model=Role)
async def update_role(
    role_id: str,
    role_update: RoleUpdate,
    current_user: User = Depends(get_current_active_user),
):
    """Update a role"""
    # Check user has permission
    check_permission(current_user.role, "admin:roles:update")
    
    # Only system admins can update roles
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update roles"
        )
    
    try:
        # In a real implementation, you would update in database
        # For demonstration, we'll update mock data
        
        # Mock role data based on ID
        if role_id == "role-1":
            role = Role(
                id="role-1",
                name="admin",
                description="System administrator with full access",
                permissions=["admin:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        elif role_id == "role-2":
            role = Role(
                id="role-2",
                name="client_admin",
                description="Client administrator with limited admin access",
                permissions=["admin:client:*", "admin:users:*", "queries:*", "visualizations:*", "analytics:*", "database:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Cannot update system roles
        if role.is_system_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot modify system roles"
            )
        
        # Apply updates
        update_data = {k: v for k, v in role_update.dict(exclude_unset=True).items()}
        for key, value in update_data.items():
            setattr(role, key, value)
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="update",
            resource="role",
            resource_id=role_id,
            details=update_data
        )
        
        logger.info(f"Updated role: {role.name}")
        return role
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating role: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating role: {str(e)}"
        )

@router.delete("/roles/{role_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_role(
    role_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Delete a role"""
    # Check user has permission
    check_permission(current_user.role, "admin:roles:delete")
    
    # Only system admins can delete roles
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete roles"
        )
    
    try:
        # In a real implementation, you would delete from database
        # For demonstration, we'll check mock data
        
        # Mock role data based on ID
        if role_id == "role-1":
            role = Role(
                id="role-1",
                name="admin",
                description="System administrator with full access",
                permissions=["admin:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        elif role_id == "role-2":
            role = Role(
                id="role-2",
                name="client_admin",
                description="Client administrator with limited admin access",
                permissions=["admin:client:*", "admin:users:*", "queries:*", "visualizations:*", "analytics:*", "database:*"],
                is_system_role=True,
                created_at=datetime.now() - timedelta(days=30)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Role not found"
            )
        
        # Cannot delete system roles
        if role.is_system_role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete system roles"
            )
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="delete",
            resource="role",
            resource_id=role_id,
            details={"name": role.name}
        )
        
        logger.info(f"Deleted role: {role.name}")
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting role: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting role: {str(e)}"
        )

@router.get("/audit-logs", response_model=List[AuditLog])
async def get_audit_logs(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    resource: Optional[str] = None,
    client_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user),
):
    """Get audit logs with filtering"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:audit:view")
    else:
        # Client admins can only view their own client's audit logs
        check_permission(current_user.role, "admin:client:audit:view")
        client_id = current_user.client_id
    
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = date.today() + timedelta(days=1)  # Include today
        if not start_date:
            start_date = end_date - timedelta(days=7)  # Last 7 days by default
        
        # Get audit logs
        logs = await get_audit_logs(
            start_date=start_date,
            end_date=end_date,
            user_id=user_id,
            action=action,
            resource=resource,
            client_id=client_id,
            limit=limit
        )
        
        logger.info(f"Retrieved audit logs, count: {len(logs)}")
        return logs
        
    except Exception as e:
        logger.error(f"Error retrieving audit logs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving audit logs: {str(e)}"
        )

@router.get("/models", response_model=Dict[str, Any])
async def get_models(
    current_user: User = Depends(get_current_active_user),
):
    """Get information about available LLM models"""
    # Check user has permission
    check_permission(current_user.role, "admin:models:view")
    
    try:
        # Get available models
        models = get_available_models()
        
        # Get active model
        active_model = get_active_model()
        
        response = {
            "available_models": models,
            "active_model": active_model.name if active_model else None
        }
        
        logger.info(f"Retrieved model information")
        return response
        
    except Exception as e:
        logger.error(f"Error retrieving model information: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving model information: {str(e)}"
        )

@router.post("/models/active", response_model=Dict[str, str])
async def set_active_model_endpoint(
    model_name: str,
    current_user: User = Depends(get_current_active_user),
):
    """Set the active LLM model"""
    # Check user has permission
    check_permission(current_user.role, "admin:models:update")
    
    # Only system admins can change models
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to change models"
        )
    
    try:
        # Get available models
        models = get_available_models()
        
        # Check if model exists
        if model_name not in [m["name"] for m in models]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model {model_name} not found"
            )
        
        # Set active model
        success = set_active_model(model_name)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to set active model to {model_name}"
            )
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="update",
            resource="model",
            details={"active_model": model_name}
        )
        
        logger.info(f"Set active model to {model_name}")
        return {"active_model": model_name}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting active model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error setting active model: {str(e)}"
        )

@router.get("/database/sync-status", response_model=List[SyncStatus])
async def get_database_sync_status(
    client_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Get database synchronization status"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:database:view")
    else:
        # Client admins can only view their own client's sync status
        check_permission(current_user.role, "admin:client:database:view")
        client_id = current_user.client_id
    
    try:
        # Get sync status
        status_list = await get_sync_status(
            client_id=client_id,
            connection_id=connection_id
        )
        
        logger.info(f"Retrieved sync status, count: {len(status_list)}")
        return status_list
        
    except Exception as e:
        logger.error(f"Error retrieving sync status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving sync status: {str(e)}"
        )

@router.post("/database/trigger-sync", response_model=Dict[str, Any])
async def trigger_database_sync(
    client_id: str,
    connection_id: str,
    force_full_sync: bool = False,
    current_user: User = Depends(get_current_active_user),
):
    """Trigger database synchronization"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:database:update")
    else:
        # Client admins can only trigger sync for their own client
        check_permission(current_user.role, "admin:client:database:update")
        if client_id != current_user.client_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to sync this client's database"
            )
    
    try:
        # Trigger sync
        sync_id = await trigger_sync(
            client_id=client_id,
            connection_id=connection_id,
            force_full_sync=force_full_sync
        )
        
        # Log audit
        from app.security.audit_logger import log_audit
        await log_audit(
            user_id=current_user.id,
            action="sync",
            resource="database",
            client_id=client_id,
            details={"connection_id": connection_id, "force_full_sync": force_full_sync}
        )
        
        logger.info(f"Triggered database sync for client: {client_id}, connection: {connection_id}")
        return {
            "sync_id": sync_id,
            "status": "initiated",
            "client_id": client_id,
            "connection_id": connection_id,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error triggering sync: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error triggering sync: {str(e)}"
        )

@router.get("/domain-mappings", response_model=List[DomainMapping])
async def get_domain_mappings_endpoint(
    client_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Get domain concept mappings for client schema"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:schema:view")
    else:
        # Client admins can only view their own client's mappings
        check_permission(current_user.role, "admin:client:schema:view")
        client_id = current_user.client_id
    
    # Ensure client_id is provided
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Client ID required"
        )
    
    try:
        # Get domain mappings
        mappings = await get_domain_mappings(client_id)
        
        logger.info(f"Retrieved domain mappings for client: {client_id}, count: {len(mappings)}")
        return mappings
        
    except Exception as e:
        logger.error(f"Error retrieving domain mappings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving domain mappings: {str(e)}"
        )

@router.put("/domain-mappings", response_model=Dict[str, Any])
async def update_domain_mapping(
    mappings: List[DomainMapping],
    current_user: User = Depends(get_current_active_user),
):
    """Update domain concept mappings"""
    # Check permissions based on role
    if current_user.role == "admin":
        check_permission(current_user.role, "admin:schema:update")
    else:
        # Client admins can only update their own client's mappings
        check_permission(current_user.role, "admin:client:schema:update")
        
        # Ensure all mappings are for the admin's client
        if not all(m.client_id == current_user.client_id for m in mappings):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update mappings for other clients"
            )
    
    try:
        # Update each mapping
        for mapping in mappings:
            # Mark as manual override
            mapping.manual_override = True
            mapping.last_updated = datetime.now()
        
        # Save mappings
        updated_count = await update_domain_mappings(mappings)
        
        # Log audit
        from app.security.audit_logger import log_audit
        for mapping in mappings:
            await log_audit(
                user_id=current_user.id,
                action="update",
                resource="domain_mapping",
                client_id=mapping.client_id,
                details={
                    "table": mapping.custom_table,
                    "column": mapping.custom_column,
                    "concept": mapping.domain_concept
                }
            )
        
        logger.info(f"Updated domain mappings, count: {updated_count}")
        return {
            "success": True,
            "updated_count": updated_count,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error updating domain mappings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating domain mappings: {str(e)}"
        )