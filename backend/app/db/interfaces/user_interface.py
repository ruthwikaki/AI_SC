from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid


from app.db.schema.schema_discovery import get_connector_for_client
from app.utils.logger import get_logger
from app.config import get_settings


# Initialize logger
logger = get_logger(__name__)


# Get settings
settings = get_settings()


class User:
    """User model"""
    def __init__(
        self,
        id: str,
        username: str,
        email: str,
        role: str,
        is_active: bool = True,
        client_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_login: Optional[datetime] = None
    ):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        self.is_active = is_active
        self.client_id = client_id
        self.created_at = created_at or datetime.now()
        self.last_login = last_login
   
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "client_id": self.client_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }
   
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """Create from dictionary"""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except ValueError:
                created_at = datetime.now()
       
        last_login = None
        if data.get("last_login"):
            try:
                last_login = datetime.fromisoformat(data["last_login"])
            except ValueError:
                last_login = None
       
        return cls(
            id=data.get("id", ""),
            username=data.get("username", ""),
            email=data.get("email", ""),
            role=data.get("role", "user"),
            is_active=data.get("is_active", True),
            client_id=data.get("client_id"),
            created_at=created_at,
            last_login=last_login
        )


class UserInDB(User):
    """User model with password hash"""
    def __init__(
        self,
        id: str,
        username: str,
        email: str,
        role: str,
        hashed_password: str,
        is_active: bool = True,
        client_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        last_login: Optional[datetime] = None
    ):
        super().__init__(
            id=id,
            username=username,
            email=email,
            role=role,
            is_active=is_active,
            client_id=client_id,
            created_at=created_at,
            last_login=last_login
        )
        self.hashed_password = hashed_password
   
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = super().to_dict()
        data["hashed_password"] = self.hashed_password
        return data
   
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserInDB':
        """Create from dictionary"""
        created_at = None
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except ValueError:
                created_at = datetime.now()
       
        last_login = None
        if data.get("last_login"):
            try:
                last_login = datetime.fromisoformat(data["last_login"])
            except ValueError:
                last_login = None
       
        return cls(
            id=data.get("id", ""),
            username=data.get("username", ""),
            email=data.get("email", ""),
            role=data.get("role", "user"),
            hashed_password=data.get("hashed_password", ""),
            is_active=data.get("is_active", True),
            client_id=data.get("client_id"),
            created_at=created_at,
            last_login=last_login
        )


class UserInterface:
    """Interface for user-related database operations"""
   
    def __init__(self, client_id: Optional[str] = None):
        """
        Initialize the user interface.
       
        Args:
            client_id: Optional client ID for multi-tenant scenarios
        """
        self.client_id = client_id
        self.admin_db_client_id = settings.admin_db_client_id or "admin"
   
    async def get_user(self, user_id: str) -> Optional[UserInDB]:
        """
        Get a user by ID.
       
        Args:
            user_id: User ID
           
        Returns:
            User or None if not found
        """
        try:
            # Get connector to admin database (which stores users)
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Query user by ID
            query = """
            SELECT id, username, email, role, hashed_password, is_active, client_id, created_at, last_login
            FROM users
            WHERE id = :user_id
            """
           
            result = await connector.execute_query(query, {"user_id": user_id})
           
            # Convert to UserInDB object if found
            if result["data"]:
                return UserInDB.from_dict(result["data"][0])
           
            return None
           
        except Exception as e:
            logger.error(f"Error retrieving user: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """
        Get a user by username.
       
        Args:
            username: Username
           
        Returns:
            User or None if not found
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Query user by username
            query = """
            SELECT id, username, email, role, hashed_password, is_active, client_id, created_at, last_login
            FROM users
            WHERE username = :username
            """
           
            result = await connector.execute_query(query, {"username": username})
           
            # Convert to UserInDB object if found
            if result["data"]:
                return UserInDB.from_dict(result["data"][0])
           
            return None
           
        except Exception as e:
            logger.error(f"Error retrieving user by username: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """
        Get a user by email.
       
        Args:
            email: Email address
           
        Returns:
            User or None if not found
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Query user by email
            query = """
            SELECT id, username, email, role, hashed_password, is_active, client_id, created_at, last_login
            FROM users
            WHERE email = :email
            """
           
            result = await connector.execute_query(query, {"email": email})
           
            # Convert to UserInDB object if found
            if result["data"]:
                return UserInDB.from_dict(result["data"][0])
           
            return None
           
        except Exception as e:
            logger.error(f"Error retrieving user by email: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        role: str = "user",
        client_id: Optional[str] = None,
        is_active: bool = True
    ) -> Optional[User]:
        """
        Create a new user.
       
        Args:
            username: Username
            email: Email address
            hashed_password: Hashed password
            role: User role
            client_id: Optional client ID
            is_active: Whether the user is active
           
        Returns:
            Created user or None if error
        """
        try:
            # If client_id not provided, use the interface client_id
            if not client_id:
                client_id = self.client_id
           
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Generate a new ID
            user_id = str(uuid.uuid4())
           
            # Insert new user
            query = """
            INSERT INTO users (id, username, email, hashed_password, role, client_id, is_active, created_at)
            VALUES (:id, :username, :email, :hashed_password, :role, :client_id, :is_active, :created_at)
            """
           
            created_at = datetime.now()
           
            params = {
                "id": user_id,
                "username": username,
                "email": email,
                "hashed_password": hashed_password,
                "role": role,
                "client_id": client_id,
                "is_active": is_active,
                "created_at": created_at.isoformat()
            }
           
            await connector.execute_query(query, params)
           
            # Return the created user
            return User(
                id=user_id,
                username=username,
                email=email,
                role=role,
                is_active=is_active,
                client_id=client_id,
                created_at=created_at
            )
           
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """
        Update a user.
       
        Args:
            user_id: User ID
            update_data: Data to update
           
        Returns:
            Updated user or None if error
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Build update query
            query_parts = []
            params = {"user_id": user_id}
           
            allowed_fields = {
                "email", "hashed_password", "role", "is_active",
                "client_id", "last_login"
            }
           
            for field, value in update_data.items():
                if field in allowed_fields:
                    query_parts.append(f"{field} = :{field}")
                    params[field] = value
           
            if not query_parts:
                # Nothing to update
                return await self.get_user(user_id)
           
            # Add updated_at field
            query_parts.append("updated_at = :updated_at")
            params["updated_at"] = datetime.now().isoformat()
           
            # Build the final query
            query = f"""
            UPDATE users
            SET {", ".join(query_parts)}
            WHERE id = :user_id
            """
           
            await connector.execute_query(query, params)
           
            # Return the updated user
            return await self.get_user(user_id)
           
        except Exception as e:
            logger.error(f"Error updating user: {str(e)}")
            return None
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
       
        Args:
            user_id: User ID
           
        Returns:
            True if deleted, False if error
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Delete the user
            query = "DELETE FROM users WHERE id = :user_id"
           
            await connector.execute_query(query, {"user_id": user_id})
           
            return True
           
        except Exception as e:
            logger.error(f"Error deleting user: {str(e)}")
            return False
        finally:
            if locals().get("connector"):
                await connector.close()
   
async def get_users(
        self,
        client_id: Optional[str] = None,
        role: Optional[str] = None,
        is_active: Optional[bool] = None
    ) -> List[User]:
        """
        Get users with optional filtering.
       
        Args:
            client_id: Optional client ID filter
            role: Optional role filter
            is_active: Optional active status filter
           
        Returns:
            List of users
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Build query with filters
            query = """
            SELECT id, username, email, role, is_active, client_id, created_at, last_login
            FROM users
            WHERE 1=1
            """
           
            params = {}
           
            if client_id:
                query += " AND client_id = :client_id"
                params["client_id"] = client_id
           
            if role:
                query += " AND role = :role"
                params["role"] = role
           
            if is_active is not None:
                query += " AND is_active = :is_active"
                params["is_active"] = is_active
           
            # Add order by
            query += " ORDER BY username"
           
            result = await connector.execute_query(query, params)
           
            # Convert to User objects
            users = [User.from_dict(row) for row in result["data"]]
           
            return users
           
        except Exception as e:
            logger.error(f"Error retrieving users: {str(e)}")
            return []
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def update_last_login(self, user_id: str) -> bool:
        """
        Update a user's last login time.
       
        Args:
            user_id: User ID
           
        Returns:
            True if updated, False if error
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Update last login time
            query = """
            UPDATE users
            SET last_login = :last_login
            WHERE id = :user_id
            """
           
            params = {
                "user_id": user_id,
                "last_login": datetime.now().isoformat()
            }
           
            await connector.execute_query(query, params)
           
            return True
           
        except Exception as e:
            logger.error(f"Error updating last login: {str(e)}")
            return False
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_saved_queries(
        self,
        user_id: str,
        client_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get saved queries for a user.
       
        Args:
            user_id: User ID
            client_id: Optional client ID
           
        Returns:
            List of saved queries
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Build query
            query = """
            SELECT id, name, description, query, created_by, created_at,
                   last_used, use_count, is_public, tags, client_id
            FROM saved_queries
            WHERE created_by = :user_id
            """
           
            params = {"user_id": user_id}
           
            if client_id:
                query += " AND client_id = :client_id"
                params["client_id"] = client_id
           
            # Add shared queries if client_id provided
            if client_id:
                query += """
                UNION
                SELECT id, name, description, query, created_by, created_at,
                       last_used, use_count, is_public, tags, client_id
                FROM saved_queries
                WHERE is_public = true AND client_id = :client_id
                """
           
            # Add order by
            query += " ORDER BY name"
           
            result = await connector.execute_query(query, params)
           
            # Process results
            saved_queries = []
            for row in result["data"]:
                # Convert tags from string to list if needed
                tags = row.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except:
                        tags = []
               
                # Convert dates
                created_at = row.get("created_at")
                if created_at and isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except:
                        created_at = None
               
                last_used = row.get("last_used")
                if last_used and isinstance(last_used, str):
                    try:
                        last_used = datetime.fromisoformat(last_used)
                    except:
                        last_used = None
               
                # Create query object
                query_obj = {
                    "id": row.get("id"),
                    "name": row.get("name"),
                    "description": row.get("description"),
                    "query": row.get("query"),
                    "created_by": row.get("created_by"),
                    "created_at": created_at,
                    "last_used": last_used,
                    "use_count": row.get("use_count", 0),
                    "is_public": row.get("is_public", False),
                    "tags": tags,
                    "client_id": row.get("client_id")
                }
               
                saved_queries.append(query_obj)
           
            return saved_queries
           
        except Exception as e:
            logger.error(f"Error retrieving saved queries: {str(e)}")
            return []
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def save_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save a query.
       
        Args:
            query_data: Query data
           
        Returns:
            Saved query data
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Check if query exists
            existing_query = None
            if query_data.get("id"):
                check_query = """
                SELECT id FROM saved_queries WHERE id = :id
                """
                check_result = await connector.execute_query(check_query, {"id": query_data["id"]})
                if check_result["data"]:
                    existing_query = query_data["id"]
           
            # Process tags
            tags = query_data.get("tags", [])
            if not isinstance(tags, str):
                tags = json.dumps(tags)
           
            if existing_query:
                # Update existing query
                update_query = """
                UPDATE saved_queries
                SET name = :name,
                    description = :description,
                    query = :query,
                    is_public = :is_public,
                    tags = :tags,
                    client_id = :client_id
                WHERE id = :id
                """
               
                params = {
                    "id": query_data["id"],
                    "name": query_data.get("name"),
                    "description": query_data.get("description"),
                    "query": query_data.get("query"),
                    "is_public": query_data.get("is_public", False),
                    "tags": tags,
                    "client_id": query_data.get("client_id")
                }
               
                await connector.execute_query(update_query, params)
               
                return query_data
               
            else:
                # Insert new query
                insert_query = """
                INSERT INTO saved_queries (
                    id, name, description, query, created_by, created_at,
                    use_count, is_public, tags, client_id
                )
                VALUES (
                    :id, :name, :description, :query, :created_by, :created_at,
                    :use_count, :is_public, :tags, :client_id
                )
                """
               
                # Generate ID if not provided
                if not query_data.get("id"):
                    query_data["id"] = str(uuid.uuid4())
               
                # Ensure created_at is set
                if not query_data.get("created_at"):
                    query_data["created_at"] = datetime.now()
               
                # Convert created_at to string if it's a datetime
                if isinstance(query_data.get("created_at"), datetime):
                    query_data["created_at"] = query_data["created_at"].isoformat()
               
                params = {
                    "id": query_data["id"],
                    "name": query_data.get("name"),
                    "description": query_data.get("description"),
                    "query": query_data.get("query"),
                    "created_by": query_data.get("created_by"),
                    "created_at": query_data.get("created_at"),
                    "use_count": query_data.get("use_count", 0),
                    "is_public": query_data.get("is_public", False),
                    "tags": tags,
                    "client_id": query_data.get("client_id")
                }
               
                await connector.execute_query(insert_query, params)
               
                return query_data
           
        except Exception as e:
            logger.error(f"Error saving query: {str(e)}")
            return query_data  # Return original data on error
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def delete_saved_query(self, query_id: str, user_id: str) -> bool:
        """
        Delete a saved query.
       
        Args:
            query_id: Query ID
            user_id: User ID (for authorization)
           
        Returns:
            True if deleted, False if error or not authorized
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Delete the query (only if created by the user)
            query = """
            DELETE FROM saved_queries
            WHERE id = :query_id AND created_by = :user_id
            """
           
            result = await connector.execute_query(query, {
                "query_id": query_id,
                "user_id": user_id
            })
           
            # PostgreSQL returns number of rows affected
            return result.get("row_count", 0) > 0
           
        except Exception as e:
            logger.error(f"Error deleting saved query: {str(e)}")
            return False
        finally:
            if locals().get("connector"):
                await connector.close()
   
    async def get_query_history(
        self,
        user_id: str,
        client_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get query execution history for a user.
       
        Args:
            user_id: User ID
            client_id: Optional client ID
            limit: Maximum number of items to return
           
        Returns:
            List of query history items
        """
        try:
            # Get connector to admin database
            connector = await get_connector_for_client(self.admin_db_client_id)
           
            # Build query
            query = """
            SELECT query_id, natural_query, sql, execution_time, timestamp, model_used
            FROM query_history
            WHERE user_id = :user_id
            """
           
            params = {"user_id": user_id}
           
            if client_id:
                query += " AND client_id = :client_id"
                params["client_id"] = client_id
           
            # Add order by and limit
            query += f" ORDER BY timestamp DESC LIMIT {limit}"
           
            result = await connector.execute_query(query, params)
           
            # Process results
            history = []
            for row in result["data"]:
                # Convert timestamp
                timestamp = row.get("timestamp")
                if timestamp and isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except:
                        timestamp = None
               
                # Create history item
                item = {
                    "query_id": row.get("query_id"),
                    "natural_query": row.get("natural_query"),
                    "sql": row.get("sql"),
                    "execution_time": row.get("execution_time"),
                    "timestamp": timestamp,
                    "model_used": row.get("model_used")
                }
               
                history.append(item)
           
            return history
           
        except Exception as e:
            logger.error(f"Error retrieving query history: {str(e)}")
            return []
        finally:
            if locals().get("connector"):
                await connector.close()