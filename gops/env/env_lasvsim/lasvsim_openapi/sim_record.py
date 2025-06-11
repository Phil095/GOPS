"""
Simulation record module for the lasvsim API.
"""
from typing import Optional

from lasvsim_openapi.http_client import HttpClient
from lasvsim_openapi.sim_record_model import (
    GetRecordIdsReq,
    GetRecordIdsRes,
    GetTrackResultsReq,
    GetTrackResultsRes,
    GetSensorResultsReq,
    GetSensorResultsRes,
    GetStepResultsReq,
    GetStepResultsRes,
    GetPathResultsReq,
    GetPathResultsRes,
    GetReferenceLineResultsReq,
    GetReferenceLineResultsRes,
)
from lasvsim_openapi.sim_record_fast import SimRecordFast


class SimRecord:
    """Simulation record client for the API."""
    sim_record_fast: SimRecordFast

    def __init__(self, http_client: HttpClient):
        """Initialize simulation record client.
        
        Args:
            http_client: HTTP client instance
        """
        self.sim_record_fast =SimRecordFast(http_client=http_client)

    def get_record_ids(self, scen_id: str, scen_ver: str) -> GetRecordIdsRes:
        """Get record IDs.
        
        Args:
            scen_id: Scenario ID
            scen_ver: Scenario version
            
        Returns:
            Record IDs response
            
        Raises:
            APIError: If the request fails
        """
        reply = self.sim_record_fast.get_record_ids(scen_id, scen_ver)
        return GetRecordIdsRes.from_dict(reply)

    def get_track_results(self, id: str, obj_id: str) -> GetTrackResultsRes:
        """Get track results.
        
        Args:
            id: Record ID
            obj_id: Object ID
            
        Returns:
            Get track results response
            
        Raises:
            APIError: If the request fails
        """
        reply = self.sim_record_fast.get_track_results(id, obj_id)
        return GetTrackResultsRes.from_dict(reply)

    def get_sensor_results(self, id: str, obj_id: str) -> GetSensorResultsRes:
        """Get sensor results.
        
        Args:
            id: Record ID
            obj_id: Object ID
            
        Returns:
            Sensor results response
            
        Raises:
            APIError: If the request fails
        """
        reply = self.sim_record_fast.get_sensor_results(id, obj_id)
        return GetSensorResultsRes.from_dict(reply)

    def get_step_results(self, id: str, obj_id: str) -> GetStepResultsRes:
        """Get step results.
        
        Args:
            id: Record ID
            obj_id: Object ID
            
        Returns:
            Step results response
            
        Raises:
            APIError: If the request fails
        """
        reply = self.sim_record_fast.get_step_results(id, obj_id)
        return GetStepResultsRes.from_dict(reply)

    def get_path_results(self, id: str, obj_id: str) -> GetPathResultsRes:
        """Get path results.
        
        Args:
            id: Record ID
            obj_id: Object ID
            
        Returns:
            Get path results response
            
        Raises:
            APIError: If the request fails
        """
        reply = self.sim_record_fast.get_path_results(id, obj_id)
        return GetPathResultsRes.from_dict(reply)

    def get_reference_line_results(self, id: str, obj_id: str) -> GetReferenceLineResultsRes:
        """Get reference line results.
        
        Args:
            id: Record ID
            obj_id: Object ID
            
        Returns:
            Get reference line results response
            
        Raises:
            APIError: If the request fails
        """
        reply = self.sim_record_fast.get_reference_line_results(id, obj_id)
        return GetReferenceLineResultsRes.from_dict(reply)
