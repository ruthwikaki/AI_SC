"""
Route Optimizer Module

This module provides functionality for optimizing delivery routes
based on various constraints and objectives.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import json
import random
from itertools import permutations

from app.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

class RouteOptimizer:
    """
    Optimizes delivery routes based on various constraints and objectives.
    """
    
    # Distance calculation methods
    DISTANCE_METHODS = {
        "euclidean": "Euclidean (straight-line) distance",
        "manhattan": "Manhattan (city block) distance",
        "haversine": "Haversine (great-circle) distance for geographic coordinates"
    }
    
    # Optimization algorithms
    ALGORITHMS = {
        "exact": "Exact solution (suitable for small problems)",
        "nearest_neighbor": "Nearest Neighbor heuristic",
        "savings": "Clarke-Wright Savings algorithm",
        "genetic": "Genetic Algorithm",
        "tabu_search": "Tabu Search meta-heuristic",
        "simulated_annealing": "Simulated Annealing meta-heuristic"
    }
    
    def __init__(
        self,
        distance_method: str = "haversine",
        algorithm: str = "nearest_neighbor"
    ):
        """
        Initialize the route optimizer.
        
        Args:
            distance_method: Method for calculating distances
            algorithm: Optimization algorithm to use
        """
        # Validate distance method
        if distance_method not in self.DISTANCE_METHODS:
            logger.warning(f"Unknown distance method: {distance_method}. Using haversine instead.")
            distance_method = "haversine"
        
        # Validate algorithm
        if algorithm not in self.ALGORITHMS:
            logger.warning(f"Unknown optimization algorithm: {algorithm}. Using nearest_neighbor instead.")
            algorithm = "nearest_neighbor"
        
        self.distance_method = distance_method
        self.algorithm = algorithm
    
    async def optimize_route(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
        optimization_objective: str = "distance",
        return_to_origin: bool = True,
        max_destinations: int = 30
    ) -> Dict[str, Any]:
        """
        Optimize a delivery route.
        
        Args:
            origin: Origin point (dict with lat, lng)
            destinations: List of destination points (each with lat, lng)
            constraints: Optional constraints (time windows, vehicle capacity, etc.)
            optimization_objective: Objective function ("distance", "time", "cost")
            return_to_origin: Whether to return to the origin point
            max_destinations: Maximum number of destinations to consider
            
        Returns:
            Dictionary with optimized route
        """
        try:
            # Validate inputs
            if not origin:
                return {"error": "No origin provided"}
            
            if not destinations:
                return {"error": "No destinations provided"}
            
            # Limit the number of destinations for performance
            if len(destinations) > max_destinations:
                logger.warning(f"Too many destinations ({len(destinations)}). Limiting to {max_destinations}.")
                destinations = destinations[:max_destinations]
            
            # Set default constraints if not provided
            if not constraints:
                constraints = {}
            
            # Preprocess locations
            origin, destinations = self._preprocess_locations(origin, destinations)
            
            # Calculate distance matrix
            distance_matrix = self._calculate_distance_matrix(origin, destinations)
            
            # Build time matrix if needed
            if optimization_objective in ("time", "cost") or "time_windows" in constraints:
                time_matrix = self._calculate_time_matrix(distance_matrix, constraints.get("speeds", {}))
            else:
                time_matrix = None
            
            # Apply the selected optimization algorithm
            if self.algorithm == "exact":
                route, metrics = self._optimize_exact(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=optimization_objective,
                    return_to_origin=return_to_origin
                )
            
            elif self.algorithm == "nearest_neighbor":
                route, metrics = self._optimize_nearest_neighbor(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=optimization_objective,
                    return_to_origin=return_to_origin
                )
            
            elif self.algorithm == "savings":
                route, metrics = self._optimize_savings(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=optimization_objective,
                    return_to_origin=return_to_origin
                )
            
            else:
                # Default to nearest neighbor
                route, metrics = self._optimize_nearest_neighbor(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=optimization_objective,
                    return_to_origin=return_to_origin
                )
            
            # Calculate route times if not already done
            if "times" not in metrics and time_matrix is not None:
                route_times = self._calculate_route_times(
                    route=route,
                    time_matrix=time_matrix,
                    constraints=constraints
                )
                metrics["times"] = route_times
            
            # Check constraints satisfaction
            constraints_satisfied = self._check_constraints(
                route=route,
                metrics=metrics,
                constraints=constraints
            )
            
            # Try to find alternative routes if constraints are not satisfied
            alternative_routes = []
            if not constraints_satisfied.get("all_satisfied", True):
                alternative_routes = self._find_alternative_routes(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=optimization_objective,
                    return_to_origin=return_to_origin
                )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                route=route,
                metrics=metrics,
                constraints_satisfied=constraints_satisfied,
                alternative_routes=alternative_routes
            )
            
            # Prepare result
            result = {
                "route": route,
                "metrics": metrics,
                "constraints_satisfied": constraints_satisfied,
                "alternative_routes": alternative_routes,
                "recommendations": recommendations,
                "parameters": {
                    "distance_method": self.distance_method,
                    "algorithm": self.algorithm,
                    "optimization_objective": optimization_objective,
                    "return_to_origin": return_to_origin
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error optimizing route: {str(e)}")
            return {
                "error": str(e),
                "route": [],
                "metrics": {}
            }
    
    def _preprocess_locations(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Preprocess location data to ensure consistent format.
        
        Args:
            origin: Origin point
            destinations: List of destination points
            
        Returns:
            Tuple of processed (origin, destinations)
        """
        # Process origin
        processed_origin = origin.copy()
        
        # Ensure lat/lng are floats
        if "lat" in processed_origin:
            processed_origin["lat"] = float(processed_origin["lat"])
        elif "latitude" in processed_origin:
            processed_origin["lat"] = float(processed_origin["latitude"])
            processed_origin["latitude"] = float(processed_origin["latitude"])
        
        if "lng" in processed_origin:
            processed_origin["lng"] = float(processed_origin["lng"])
        elif "lon" in processed_origin or "longitude" in processed_origin:
            lon_key = "lon" if "lon" in processed_origin else "longitude"
            processed_origin["lng"] = float(processed_origin[lon_key])
            processed_origin[lon_key] = float(processed_origin[lon_key])
        
        # Add index
        processed_origin["index"] = 0
        processed_origin["name"] = processed_origin.get("name", "Origin")
        processed_origin["type"] = "origin"
        
        # Process destinations
        processed_destinations = []
        for i, dest in enumerate(destinations):
            processed_dest = dest.copy()
            
            # Ensure lat/lng are floats
            if "lat" in processed_dest:
                processed_dest["lat"] = float(processed_dest["lat"])
            elif "latitude" in processed_dest:
                processed_dest["lat"] = float(processed_dest["latitude"])
                processed_dest["latitude"] = float(processed_dest["latitude"])
            
            if "lng" in processed_dest:
                processed_dest["lng"] = float(processed_dest["lng"])
            elif "lon" in processed_dest or "longitude" in processed_dest:
                lon_key = "lon" if "lon" in processed_dest else "longitude"
                processed_dest["lng"] = float(processed_dest[lon_key])
                processed_dest[lon_key] = float(processed_dest[lon_key])
            
            # Add index
            processed_dest["index"] = i + 1
            processed_dest["name"] = processed_dest.get("name", f"Destination {i+1}")
            processed_dest["type"] = "destination"
            
            processed_destinations.append(processed_dest)
        
        return processed_origin, processed_destinations
    
    def _calculate_distance_matrix(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]]
    ) -> List[List[float]]:
        """
        Calculate distance matrix between all locations.
        
        Args:
            origin: Origin point
            destinations: List of destination points
            
        Returns:
            Distance matrix (2D list of distances)
        """
        # Create a list of all locations (origin + destinations)
        locations = [origin] + destinations
        n = len(locations)
        
        # Initialize distance matrix
        distance_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Calculate distances
        for i in range(n):
            for j in range(i+1, n):
                # Calculate distance between locations[i] and locations[j]
                distance = self._calculate_distance(locations[i], locations[j])
                
                # Distance matrix is symmetric
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        return distance_matrix
    
    def _calculate_distance(
        self,
        loc1: Dict[str, Any],
        loc2: Dict[str, Any]
    ) -> float:
        """
        Calculate distance between two locations.
        
        Args:
            loc1: First location
            loc2: Second location
            
        Returns:
            Distance between locations
        """
        if self.distance_method == "euclidean":
            # Euclidean distance (straight-line)
            return math.sqrt(
                (loc1["lat"] - loc2["lat"])**2 + 
                (loc1["lng"] - loc2["lng"])**2
            )
        
        elif self.distance_method == "manhattan":
            # Manhattan distance (city block)
            return abs(loc1["lat"] - loc2["lat"]) + abs(loc1["lng"] - loc2["lng"])
        
        elif self.distance_method == "haversine":
            # Haversine distance (great-circle)
            earth_radius = 6371.0  # kilometers
            
            # Convert latitude and longitude from degrees to radians
            lat1 = math.radians(loc1["lat"])
            lng1 = math.radians(loc1["lng"])
            lat2 = math.radians(loc2["lat"])
            lng2 = math.radians(loc2["lng"])
            
            # Haversine formula
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = earth_radius * c
            
            return distance
        
        else:
            # Default to Euclidean distance
            return math.sqrt(
                (loc1["lat"] - loc2["lat"])**2 + 
                (loc1["lng"] - loc2["lng"])**2
            )
    
    def _calculate_time_matrix(
        self,
        distance_matrix: List[List[float]],
        speeds: Dict[str, float]
    ) -> List[List[float]]:
        """
        Calculate time matrix based on distances and speeds.
        
        Args:
            distance_matrix: Distance matrix
            speeds: Dict with speed information
            
        Returns:
            Time matrix (2D list of times in hours)
        """
        # Get default speed (50 km/h if not specified)
        default_speed = speeds.get("default", 50.0)  # km/h
        
        # Initialize time matrix with same dimensions as distance matrix
        n = len(distance_matrix)
        time_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
        
        # Calculate times
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Time = Distance / Speed
                    # Convert speed from km/h to km/min (divide by 60)
                    time_matrix[i][j] = distance_matrix[i][j] / (default_speed / 60.0)
        
        return time_matrix
    
    def _optimize_exact(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]],
        distance_matrix: List[List[float]],
        time_matrix: Optional[List[List[float]]],
        constraints: Dict[str, Any],
        objective: str,
        return_to_origin: bool
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimize route using exact algorithm (suitable for small problems).
        
        Args:
            origin: Origin point
            destinations: List of destination points
            distance_matrix: Distance matrix
            time_matrix: Time matrix
            constraints: Constraints
            objective: Optimization objective
            return_to_origin: Whether to return to origin
            
        Returns:
            Tuple of (optimized route, metrics)
        """
        try:
            # This method is only suitable for small problems
            if len(destinations) > 10:
                logger.warning("Exact algorithm is only suitable for small problems (up to 10 destinations). Falling back to nearest neighbor.")
                return self._optimize_nearest_neighbor(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=objective,
                    return_to_origin=return_to_origin
                )
            
            # Get the indices for all destinations
            dest_indices = [d["index"] for d in destinations]
            origin_idx = origin["index"]
            
            # Generate all possible permutations
            all_routes = list(permutations(dest_indices))
            
            best_route_indices = None
            best_value = float('inf')
            
            # Evaluate each permutation
            for route_perm in all_routes:
                # Create the full route (origin -> destinations -> origin if returning)
                route_indices = [origin_idx] + list(route_perm)
                if return_to_origin:
                    route_indices.append(origin_idx)
                
                # Calculate total distance
                total_distance = 0
                for i in range(len(route_indices) - 1):
                    total_distance += distance_matrix[route_indices[i]][route_indices[i+1]]
                
                # Calculate total time if needed
                total_time = 0
                if time_matrix and (objective == "time" or constraints.get("time_windows")):
                    for i in range(len(route_indices) - 1):
                        total_time += time_matrix[route_indices[i]][route_indices[i+1]]
                
                # Choose value to optimize based on objective
                if objective == "distance":
                    value = total_distance
                elif objective == "time":
                    value = total_time
                else:  # default to distance
                    value = total_distance
                
                # Check if this route is better
                if value < best_value:
                    # Check constraints
                    if self._is_route_feasible(
                        route_indices=route_indices,
                        distance_matrix=distance_matrix,
                        time_matrix=time_matrix,
                        constraints=constraints
                    ):
                        best_value = value
                        best_route_indices = route_indices
            
            # If no feasible route was found, ignore constraints
            if best_route_indices is None:
                logger.warning("No feasible route found. Ignoring constraints.")
                
                # Reevaluate each permutation without constraints
                for route_perm in all_routes:
                    # Create the full route
                    route_indices = [origin_idx] + list(route_perm)
                    if return_to_origin:
                        route_indices.append(origin_idx)
                    
                    # Calculate value based on objective
                    if objective == "distance":
                        value = sum(distance_matrix[route_indices[i]][route_indices[i+1]] 
                                   for i in range(len(route_indices) - 1))
                    elif objective == "time":
                        value = sum(time_matrix[route_indices[i]][route_indices[i+1]] 
                                   for i in range(len(route_indices) - 1))
                    else:
                        value = sum(distance_matrix[route_indices[i]][route_indices[i+1]] 
                                   for i in range(len(route_indices) - 1))
                    
                    # Check if this route is better
                    if value < best_value:
                        best_value = value
                        best_route_indices = route_indices
            
            # Convert route indices to locations
            all_locations = [origin] + destinations
            route = [all_locations[i] for i in best_route_indices]
            
            # Calculate metrics
            metrics = self._calculate_route_metrics(
                route_indices=best_route_indices,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix
            )
            
            return route, metrics
            
        except Exception as e:
            logger.error(f"Error in exact optimization: {str(e)}")
            # Fall back to nearest neighbor
            return self._optimize_nearest_neighbor(
                origin=origin,
                destinations=destinations,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                constraints=constraints,
                objective=objective,
                return_to_origin=return_to_origin
            )
    
    def _optimize_nearest_neighbor(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]],
        distance_matrix: List[List[float]],
        time_matrix: Optional[List[List[float]]],
        constraints: Dict[str, Any],
        objective: str,
        return_to_origin: bool
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimize route using Nearest Neighbor heuristic.
        
        Args:
            origin: Origin point
            destinations: List of destination points
            distance_matrix: Distance matrix
            time_matrix: Time matrix
            constraints: Constraints
            objective: Optimization objective
            return_to_origin: Whether to return to origin
            
        Returns:
            Tuple of (optimized route, metrics)
        """
        try:
            # Get the indices for all destinations
            dest_indices = [d["index"] for d in destinations]
            origin_idx = origin["index"]
            
            # Initialize route with origin
            route_indices = [origin_idx]
            unvisited = set(dest_indices)
            
            # Start from the origin
            current = origin_idx
            
            # Consider time windows if specified
            time_windows = constraints.get("time_windows", {})
            current_time = constraints.get("start_time", 0)
            
            # Build the route one step at a time
            while unvisited:
                best_next = None
                best_value = float('inf')
                
                for next_idx in unvisited:
                    # Choose value to optimize based on objective
                    if objective == "distance":
                        value = distance_matrix[current][next_idx]
                    elif objective == "time":
                        value = time_matrix[current][next_idx]
                    else:  # default to distance
                        value = distance_matrix[current][next_idx]
                    
                    # Check time window constraints if applicable
                    if time_windows and str(next_idx) in time_windows:
                        # Calculate arrival time
                        travel_time = time_matrix[current][next_idx]
                        arrival_time = current_time + travel_time
                        
                        # Get time window for this location
                        tw = time_windows[str(next_idx)]
                        earliest = tw.get("earliest", 0)
                        latest = tw.get("latest", float('inf'))
                        
                        # Check if arrival time is within the time window
                        if arrival_time > latest:
                            # Too late, skip this location
                            continue
                        
                        # If arriving early, add waiting time to the value
                        if arrival_time < earliest:
                            value += (earliest - arrival_time) if objective == "time" else 0
                    
                    # Update best next location
                    if value < best_value:
                        best_value = value
                        best_next = next_idx
                
                if best_next is None:
                    # No feasible next location found
                    break
                
                # Add the best next location to the route
                route_indices.append(best_next)
                unvisited.remove(best_next)
                
                # Update current location and time
                current = best_next
                if time_matrix:
                    travel_time = time_matrix[route_indices[-2]][current]
                    current_time += travel_time
                    
                    # Add service time if specified
                    service_time = constraints.get("service_times", {}).get(str(current), 0)
                    current_time += service_time
                    
                    # Add waiting time if arriving early to a location with time window
                    if time_windows and str(current) in time_windows:
                        earliest = time_windows[str(current)].get("earliest", 0)
                        if current_time < earliest:
                            current_time = earliest
            
            # Return to origin if required
            if return_to_origin:
                route_indices.append(origin_idx)
            
            # Convert route indices to locations
            all_locations = [origin] + destinations
            route = [all_locations[i] for i in route_indices]
            
            # Calculate metrics
            metrics = self._calculate_route_metrics(
                route_indices=route_indices,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix
            )
            
            return route, metrics
            
        except Exception as e:
            logger.error(f"Error in nearest neighbor optimization: {str(e)}")
            # Create a simple sequential route as fallback
            all_locations = [origin] + destinations
            if return_to_origin:
                all_locations.append(origin)
            
            # Calculate basic metrics
            metrics = {
                "total_distance": sum(
                    distance_matrix[all_locations[i]["index"]][all_locations[i+1]["index"]]
                    for i in range(len(all_locations) - 1)
                ),
                "total_time": sum(
                    time_matrix[all_locations[i]["index"]][all_locations[i+1]["index"]]
                    for i in range(len(all_locations) - 1)
                ) if time_matrix else None
            }
            
            return all_locations, metrics
    
    def _optimize_savings(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]],
        distance_matrix: List[List[float]],
        time_matrix: Optional[List[List[float]]],
        constraints: Dict[str, Any],
        objective: str,
        return_to_origin: bool
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Optimize route using Clarke-Wright Savings algorithm.
        
        Args:
            origin: Origin point
            destinations: List of destination points
            distance_matrix: Distance matrix
            time_matrix: Time matrix
            constraints: Constraints
            objective: Optimization objective
            return_to_origin: Whether to return to origin
            
        Returns:
            Tuple of (optimized route, metrics)
        """
        try:
            # Get the indices for all destinations
            dest_indices = [d["index"] for d in destinations]
            origin_idx = origin["index"]
            
            # Calculate savings for each pair of destinations
            savings = {}
            for i in dest_indices:
                for j in dest_indices:
                    if i != j:
                        # Choose matrix based on objective
                        matrix = distance_matrix if objective == "distance" else time_matrix
                        if not matrix:
                            matrix = distance_matrix
                        
                        # Calculate savings: d(O,i) + d(O,j) - d(i,j)
                        saving = matrix[origin_idx][i] + matrix[origin_idx][j] - matrix[i][j]
                        savings[(i, j)] = saving
            
            # Sort savings in descending order
            sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)
            
            # Initialize routes: one route per destination
            routes = [[origin_idx, i, origin_idx] for i in dest_indices]
            
            # Merge routes using savings
            for (i, j), saving in sorted_savings:
                # Find routes containing i and j
                route_i = None
                route_j = None
                for r in routes:
                    if i in r:
                        route_i = r
                    if j in r:
                        route_j = r
                
                # Check if i and j are in different routes
                if route_i and route_j and route_i != route_j:
                    # Check if i and j are at the ends of their routes
                    i_pos = route_i.index(i)
                    j_pos = route_j.index(j)
                    
                    # Only merge if i and j are at the ends
                    if ((i_pos == 1 and j_pos == len(route_j) - 2) or
                        (j_pos == 1 and i_pos == len(route_i) - 2)):
                        
                        # Merge the routes
                        if i_pos == 1 and j_pos == len(route_j) - 2:
                            # route_i = [O, i, O], route_j = [O, ..., j, O]
                            # result = [O, i, ..., j, O]
                            merged_route = [origin_idx] + route_i[1:-1] + route_j[1:]
                        else:
                            # route_i = [O, ..., i, O], route_j = [O, j, O]
                            # result = [O, j, ..., i, O]
                            merged_route = [origin_idx] + route_j[1:-1] + route_i[1:]
                        
                        # Check constraints
                        if self._is_route_feasible(
                            route_indices=merged_route,
                            distance_matrix=distance_matrix,
                            time_matrix=time_matrix,
                            constraints=constraints
                        ):
                            # Remove the original routes
                            routes.remove(route_i)
                            routes.remove(route_j)
                            
                            # Add the merged route
                            routes.append(merged_route)
            
            # For simplicity, we assume all destinations can be served in a single route
            # In a full implementation, you might need to handle multiple routes
            if len(routes) > 1:
                logger.warning(f"Multiple routes generated ({len(routes)}). Using the first route only.")
            
            # Get the first route
            route_indices = routes[0]
            
            # Remove the final return to origin if not required
            if not return_to_origin:
                route_indices = route_indices[:-1]
            
            # Convert route indices to locations
            all_locations = [origin] + destinations
            route = [all_locations[i] for i in route_indices]
            
            # Calculate metrics
            metrics = self._calculate_route_metrics(
                route_indices=route_indices,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix
            )
            
            return route, metrics
            
        except Exception as e:
            logger.error(f"Error in savings algorithm optimization: {str(e)}")
            # Fall back to nearest neighbor
            return self._optimize_nearest_neighbor(
                origin=origin,
                destinations=destinations,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                constraints=constraints,
                objective=objective,
                return_to_origin=return_to_origin
            )
    
    def _is_route_feasible(
        self,
        route_indices: List[int],
        distance_matrix: List[List[float]],
        time_matrix: Optional[List[List[float]]],
        constraints: Dict[str, Any]
    ) -> bool:
        """
        Check if a route is feasible given the constraints.
        
        Args:
            route_indices: List of location indices in the route
            distance_matrix: Distance matrix
            time_matrix: Time matrix
            constraints: Constraints
            
        Returns:
            True if the route is feasible, False otherwise
        """
        # Check time window constraints if applicable
        time_windows = constraints.get("time_windows", {})
        if time_windows and time_matrix:
            current_time = constraints.get("start_time", 0)
            
            for i in range(len(route_indices) - 1):
                current = route_indices[i]
                next_idx = route_indices[i+1]
                
                # Travel time to next location
                travel_time = time_matrix[current][next_idx]
                current_time += travel_time
                
                # Check time window for next location
                if str(next_idx) in time_windows:
                    tw = time_windows[str(next_idx)]
                    earliest = tw.get("earliest", 0)
                    latest = tw.get("latest", float('inf'))
                    
                    # If arriving early, wait until the time window opens
                    if current_time < earliest:
                        current_time = earliest
                    
                    # If arriving late, the route is infeasible
                    if current_time > latest:
                        return False
                
                # Add service time
                service_time = constraints.get("service_times", {}).get(str(next_idx), 0)
                current_time += service_time
        
        # Check vehicle capacity constraints if applicable
        capacity = constraints.get("vehicle_capacity")
        if capacity is not None:
            demands = constraints.get("demands", {})
            if demands:
                total_demand = sum(demands.get(str(idx), 0) for idx in route_indices)
                if total_demand > capacity:
                    return False
        
        # All constraints satisfied
        return True
    
    def _calculate_route_metrics(
        self,
        route_indices: List[int],
        distance_matrix: List[List[float]],
        time_matrix: Optional[List[List[float]]]
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a route.
        
        Args:
            route_indices: List of location indices in the route
            distance_matrix: Distance matrix
            time_matrix: Time matrix
            
        Returns:
            Dictionary with route metrics
        """
        # Calculate total distance
        total_distance = 0
        for i in range(len(route_indices) - 1):
            total_distance += distance_matrix[route_indices[i]][route_indices[i+1]]
        
        # Calculate total time if time matrix is available
        total_time = None
        if time_matrix:
            total_time = 0
            for i in range(len(route_indices) - 1):
                total_time += time_matrix[route_indices[i]][route_indices[i+1]]
        
        # Prepare metrics
        metrics = {
            "total_distance": total_distance,
            "total_time": total_time,
            "num_stops": len(route_indices) - 1,  # Exclude origin
            "distances": [
                distance_matrix[route_indices[i]][route_indices[i+1]]
                for i in range(len(route_indices) - 1)
            ]
        }
        
        # Add times if available
        if time_matrix:
            metrics["times"] = [
                time_matrix[route_indices[i]][route_indices[i+1]]
                for i in range(len(route_indices) - 1)
            ]
        
        return metrics
    
    def _calculate_route_times(
        self,
        route: List[Dict[str, Any]],
        time_matrix: List[List[float]],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate detailed timing information for a route.
        
        Args:
            route: Route (list of locations)
            time_matrix: Time matrix
            constraints: Constraints
            
        Returns:
            Dictionary with route timing information
        """
        # Get service times and time windows
        service_times = constraints.get("service_times", {})
        time_windows = constraints.get("time_windows", {})
        start_time = constraints.get("start_time", 0)
        
        # Initialize timing information
        route_times = {
            "departure_times": [],
            "arrival_times": [],
            "waiting_times": [],
            "service_times": []
        }
        
        current_time = start_time
        
        for i in range(len(route) - 1):
            current = route[i]
            next_loc = route[i+1]
            
            # Record departure time from current location
            route_times["departure_times"].append(current_time)
            
            # Travel time to next location
            travel_time = time_matrix[current["index"]][next_loc["index"]]
            
            # Calculate arrival time at next location
            arrival_time = current_time + travel_time
            route_times["arrival_times"].append(arrival_time)
            
            # Check time window for next location
            waiting_time = 0
            if time_windows and str(next_loc["index"]) in time_windows:
                tw = time_windows[str(next_loc["index"])]
                earliest = tw.get("earliest", 0)
                
                # If arriving early, calculate waiting time
                if arrival_time < earliest:
                    waiting_time = earliest - arrival_time
                    current_time = earliest
                else:
                    current_time = arrival_time
            else:
                current_time = arrival_time
            
            route_times["waiting_times"].append(waiting_time)
            
            # Add service time for next location
            service_time = service_times.get(str(next_loc["index"]), 0)
            route_times["service_times"].append(service_time)
            
            # Update current time
            current_time += service_time
        
        # Add departure time for the last location
        route_times["departure_times"].append(current_time)
        
        return route_times
    
    def _check_constraints(
        self,
        route: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if the route satisfies all constraints.
        
        Args:
            route: Route (list of locations)
            metrics: Route metrics
            constraints: Constraints
            
        Returns:
            Dictionary with constraint satisfaction information
        """
        constraints_satisfied = {
            "all_satisfied": True
        }
        
        # Check time window constraints
        time_windows = constraints.get("time_windows", {})
        if time_windows and "times" in metrics:
            times = metrics["times"]
            arrival_times = times.get("arrival_times", [])
            
            time_windows_satisfied = True
            violations = []
            
            for i, location in enumerate(route[1:]):  # Skip origin
                loc_idx = str(location["index"])
                if loc_idx in time_windows:
                    tw = time_windows[loc_idx]
                    latest = tw.get("latest", float('inf'))
                    
                    if i < len(arrival_times) and arrival_times[i] > latest:
                        time_windows_satisfied = False
                        violations.append({
                            "location": location["name"],
                            "arrival_time": arrival_times[i],
                            "latest_time": latest,
                            "violation": arrival_times[i] - latest
                        })
            
            constraints_satisfied["time_windows"] = {
                "satisfied": time_windows_satisfied,
                "violations": violations
            }
            
            if not time_windows_satisfied:
                constraints_satisfied["all_satisfied"] = False
        
        # Check vehicle capacity constraints
        capacity = constraints.get("vehicle_capacity")
        if capacity is not None:
            demands = constraints.get("demands", {})
            if demands:
                total_demand = sum(demands.get(str(loc["index"]), 0) for loc in route)
                capacity_satisfied = total_demand <= capacity
                
                constraints_satisfied["capacity"] = {
                    "satisfied": capacity_satisfied,
                    "total_demand": total_demand,
                    "capacity": capacity,
                    "utilization": total_demand / capacity if capacity > 0 else 0
                }
                
                if not capacity_satisfied:
                    constraints_satisfied["all_satisfied"] = False
        
        return constraints_satisfied
    
    def _find_alternative_routes(
        self,
        origin: Dict[str, Any],
        destinations: List[Dict[str, Any]],
        distance_matrix: List[List[float]],
        time_matrix: Optional[List[List[float]]],
        constraints: Dict[str, Any],
        objective: str,
        return_to_origin: bool
    ) -> List[Dict[str, Any]]:
        """
        Find alternative routes if constraints are not satisfied.
        
        Args:
            origin: Origin point
            destinations: List of destination points
            distance_matrix: Distance matrix
            time_matrix: Time matrix
            constraints: Constraints
            objective: Optimization objective
            return_to_origin: Whether to return to origin
            
        Returns:
            List of alternative routes
        """
        alternative_routes = []
        
        try:
            # 1. Try a different optimization algorithm
            if self.algorithm != "nearest_neighbor":
                alt_algorithm = "nearest_neighbor"
            else:
                alt_algorithm = "savings"
            
            optimizer = RouteOptimizer(
                distance_method=self.distance_method,
                algorithm=alt_algorithm
            )
            
            alt_route, alt_metrics = optimizer._optimize_nearest_neighbor(
                origin=origin,
                destinations=destinations,
                distance_matrix=distance_matrix,
                time_matrix=time_matrix,
                constraints=constraints,
                objective=objective,
                return_to_origin=return_to_origin
            )
            
            # Check if this alternative route is feasible
            alt_constraints_satisfied = self._check_constraints(
                route=alt_route,
                metrics=alt_metrics,
                constraints=constraints
            )
            
            if alt_constraints_satisfied.get("all_satisfied", False):
                alternative_routes.append({
                    "route": alt_route,
                    "metrics": alt_metrics,
                    "description": f"Alternative route using {alt_algorithm} algorithm"
                })
            
            # 2. Try relaxing time window constraints
            if "time_windows" in constraints:
                # Create a copy of constraints with relaxed time windows
                relaxed_constraints = constraints.copy()
                relaxed_time_windows = {}
                
                for loc_id, tw in constraints["time_windows"].items():
                    relaxed_tw = tw.copy()
                    # Extend the latest time by 30 minutes
                    if "latest" in relaxed_tw:
                        relaxed_tw["latest"] += 30  # Add 30 minutes
                    
                    relaxed_time_windows[loc_id] = relaxed_tw
                
                relaxed_constraints["time_windows"] = relaxed_time_windows
                
                # Try with relaxed constraints
                relaxed_route, relaxed_metrics = self._optimize_nearest_neighbor(
                    origin=origin,
                    destinations=destinations,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=relaxed_constraints,
                    objective=objective,
                    return_to_origin=return_to_origin
                )
                
                alternative_routes.append({
                    "route": relaxed_route,
                    "metrics": relaxed_metrics,
                    "description": "Alternative route with relaxed time windows (+30 min)"
                })
            
            # 3. Try splitting the route into multiple routes
            if len(destinations) > 5:
                # Split destinations into two groups
                mid = len(destinations) // 2
                first_group = destinations[:mid]
                second_group = destinations[mid:]
                
                # Optimize each group separately
                first_route, first_metrics = self._optimize_nearest_neighbor(
                    origin=origin,
                    destinations=first_group,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=objective,
                    return_to_origin=return_to_origin
                )
                
                second_route, second_metrics = self._optimize_nearest_neighbor(
                    origin=origin,
                    destinations=second_group,
                    distance_matrix=distance_matrix,
                    time_matrix=time_matrix,
                    constraints=constraints,
                    objective=objective,
                    return_to_origin=return_to_origin
                )
                
                # Combine metrics for the two routes
                combined_metrics = {
                    "total_distance": first_metrics["total_distance"] + second_metrics["total_distance"],
                    "total_time": (first_metrics["total_time"] + second_metrics["total_time"]) 
                                 if first_metrics["total_time"] and second_metrics["total_time"] else None,
                    "num_stops": first_metrics["num_stops"] + second_metrics["num_stops"],
                    "route_1": first_metrics,
                    "route_2": second_metrics
                }
                
                alternative_routes.append({
                    "route_1": first_route,
                    "route_2": second_route,
                    "metrics": combined_metrics,
                    "description": "Split into two separate routes"
                })
            
            return alternative_routes
            
        except Exception as e:
            logger.error(f"Error finding alternative routes: {str(e)}")
            return []
    
    def _generate_recommendations(
        self,
        route: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        constraints_satisfied: Dict[str, Any],
        alternative_routes: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate recommendations based on the route and metrics.
        
        Args:
            route: Optimized route
            metrics: Route metrics
            constraints_satisfied: Constraint satisfaction information
            alternative_routes: Alternative routes
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Add recommendations based on route characteristics
            if len(route) > 10:
                recommendations.append(
                    f"Route has {len(route)} stops. Consider splitting into multiple routes for better manageability."
                )
            
            # Add recommendations based on time windows
            if not constraints_satisfied.get("all_satisfied", True):
                if "time_windows" in constraints_satisfied and not constraints_satisfied["time_windows"]["satisfied"]:
                    recommendations.append(
                        "Time window constraints are not satisfied. Consider adjusting delivery schedules or using multiple vehicles."
                    )
                
                if "capacity" in constraints_satisfied and not constraints_satisfied["capacity"]["satisfied"]:
                    utilization = constraints_satisfied["capacity"]["utilization"]
                    recommendations.append(
                        f"Vehicle capacity exceeded ({utilization:.1%} utilization). Consider using a larger vehicle or multiple vehicles."
                    )
            
            # Add recommendations based on route efficiency
            if metrics.get("total_distance") and len(route) > 3:
                avg_distance_per_stop = metrics["total_distance"] / (len(route) - 1)
                
                if avg_distance_per_stop > 20:  # km
                    recommendations.append(
                        f"High average distance between stops ({avg_distance_per_stop:.1f} km). Consider grouping deliveries by geographic area."
                    )
            
            # Add recommendations based on alternative routes
            if alternative_routes:
                for alt_route in alternative_routes:
                    if "route_1" in alt_route:
                        # This is a split route
                        recommendations.append(
                            "Splitting the route into multiple trips may improve efficiency and help satisfy constraints."
                        )
                        break
            
            # Add general recommendations
            recommendations.append(
                "Regularly review and optimize routes as demand patterns change."
            )
            
            recommendations.append(
                "Consider traffic patterns and peak travel times when scheduling deliveries."
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]

    @staticmethod
    async def get_location_data(
        client_id: str,
        location_ids: List[str] = None,
        location_type: str = "delivery",
        connection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get location data for route optimization.
        
        Args:
            client_id: Client ID
            location_ids: Optional list of location IDs
            location_type: Type of locations to retrieve
            connection_id: Optional connection ID
            
        Returns:
            Dictionary with location data
        """
        try:
            # Get data from database
            from app.db.interfaces.location_interface import LocationInterface
            
            # Create interface
            location_interface = LocationInterface(client_id=client_id, connection_id=connection_id)
            
            # Get locations
            locations = await location_interface.get_locations(
                location_ids=location_ids,
                location_type=location_type
            )
            
            # Separate origin and destinations
            origin = next((loc for loc in locations if loc.get("type") == "origin"), None)
            destinations = [loc for loc in locations if loc.get("type") == "destination"]
            
            # If no explicit origin, use the first warehouse or distribution center
            if not origin:
                origin = next(
                    (loc for loc in locations if loc.get("location_type") in ["warehouse", "distribution_center"]), 
                    None
                )
            
            # If still no origin, use the first location as origin
            if not origin and locations:
                origin = locations[0]
                destinations = locations[1:]
            
            return {
                "origin": origin,
                "destinations": destinations,
                "all_locations": locations
            }
            
        except Exception as e:
            logger.error(f"Error getting location data: {str(e)}")
            
            # Generate mock data for demonstration or testing
            origin, destinations = RouteOptimizer._generate_mock_location_data(
                location_type=location_type
            )
            
            logger.warning(f"Using mock location data: 1 origin, {len(destinations)} destinations")
            return {
                "origin": origin,
                "destinations": destinations,
                "all_locations": [origin] + destinations,
                "is_mock_data": True
            }
    
    @staticmethod
    def _generate_mock_location_data(
        location_type: str = "delivery",
        num_destinations: int = 10,
        base_lat: float = 40.7128,  # New York City
        base_lng: float = -74.0060
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate mock location data for testing.
        
        Args:
            location_type: Type of locations to generate
            num_destinations: Number of destinations to generate
            base_lat: Base latitude
            base_lng: Base longitude
            
        Returns:
            Tuple of (origin, destinations)
        """
        # Generate origin
        origin = {
            "id": "origin-1",
            "name": "Distribution Center",
            "type": "origin",
            "location_type": "distribution_center",
            "lat": base_lat,
            "lng": base_lng,
            "address": "123 Main St, New York, NY 10001",
            "is_mock_data": True
        }
        
        # Generate destinations
        destinations = []
        for i in range(num_destinations):
            # Generate random coordinates within ~10km of the origin
            lat_offset = (random.random() - 0.5) * 0.2  # +/- 0.1 degrees (~11km)
            lng_offset = (random.random() - 0.5) * 0.2  # +/- 0.1 degrees (~11km at equator, less at higher latitudes)
            
            dest = {
                "id": f"dest-{i+1}",
                "name": f"Customer {i+1}",
                "type": "destination",
                "location_type": location_type,
                "lat": base_lat + lat_offset,
                "lng": base_lng + lng_offset,
                "address": f"{100+i} Delivery St, New York, NY 10001",
                "is_mock_data": True
            }
            
            destinations.append(dest)
        
        return origin, destinations