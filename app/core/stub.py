"""
Openfabric Application Stub - Remote Application Interface

This module provides a lightweight client interface (Stub) for connecting to and
interacting with multiple Openfabric applications. The Stub manages connections,
schemas, and manifests for remote applications, enabling seamless execution of
distributed AI services.

Key Features:
- Dynamic manifest and schema loading from remote applications
- WebSocket-based real-time communication via Remote connections
- Automatic resource resolution for binary data handling
- Type-safe schema validation for input/output data
- Multi-application support with unified interface

The Stub class serves as the primary interface for the AI Creative Pipeline to
communicate with Openfabric's text-to-image and image-to-3D applications.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Tuple

import requests

from core.remote import Remote
from openfabric_pysdk.helper import has_resource_fields, json_schema_to_marshmallow, resolve_resources
from openfabric_pysdk.loader import OutputSchemaInst

# Type aliases for improved code clarity and maintainability
Manifests = Dict[str, dict]  # Maps app IDs to their manifest data
Schemas = Dict[str, Tuple[dict, dict]]  # Maps app IDs to (input_schema, output_schema) tuples
Connections = Dict[str, Remote]  # Maps app IDs to their WebSocket Remote connections


class Stub:
    """
    Stub acts as a lightweight client interface for connecting to Openfabric applications.
    
    This class provides a unified interface for managing multiple remote Openfabric
    applications by handling their manifests, schemas, and WebSocket connections.
    It enables the AI Creative Pipeline to seamlessly communicate with distributed
    AI services such as text-to-image and image-to-3D conversion applications.
    
    The Stub automatically handles:
    - Manifest loading for application metadata
    - Input/output schema validation
    - WebSocket connection management
    - Resource resolution for binary data (images, models)
    - Error handling and logging for remote operations
    
    Attributes:
        _schema (Schemas): Dictionary mapping app IDs to their input/output schemas
        _manifest (Manifests): Dictionary mapping app IDs to their manifest metadata
        _connections (Connections): Dictionary mapping app IDs to their Remote connections
    """

    def __init__(self, app_ids: List[str]):
        """
        Initialize the Stub instance by establishing connections to specified applications.
        
        For each application ID provided, this method:
        1. Fetches the application manifest containing metadata
        2. Retrieves input and output schemas for data validation
        3. Establishes a WebSocket connection via Remote class
        4. Stores all information for future use
        
        Args:
            app_ids: List of application identifiers (hostnames or URLs)
                    for Openfabric applications to connect to
        """
        self._schema: Schemas = {}
        self._manifest: Manifests = {}
        self._connections: Connections = {}

        for app_id in app_ids:
            base_url = app_id.strip('/')

            try:
                # Fetch application manifest containing metadata and capabilities
                manifest = requests.get(f"https://{base_url}/manifest", timeout=5).json()
                logging.info(f"[{app_id}] Manifest loaded: {manifest}")
                self._manifest[app_id] = manifest

                # Fetch input schema for request validation
                input_schema = requests.get(f"https://{base_url}/schema?type=input", timeout=5).json()
                logging.info(f"[{app_id}] Input schema loaded: {input_schema}")

                # Fetch output schema for response validation
                output_schema = requests.get(f"https://{base_url}/schema?type=output", timeout=5).json()
                logging.info(f"[{app_id}] Output schema loaded: {output_schema}")
                self._schema[app_id] = (input_schema, output_schema)

                # Establish WebSocket connection for real-time communication
                self._connections[app_id] = Remote(f"wss://{base_url}/app", f"{app_id}-proxy").connect()
                logging.info(f"[{app_id}] Connection established.")
            except Exception as e:
                logging.error(f"[{app_id}] Initialization failed: {e}")

    def call(self, app_id: str, data: Any, uid: str = 'super-user') -> dict:
        """
        Execute a request to the specified Openfabric application.
        
        This method sends input data to a remote application via WebSocket connection,
        waits for the response, and handles any necessary resource resolution for
        binary data such as images or 3D models.
        
        Args:
            app_id: The application ID to route the request to
            data: The input data to send to the application (must conform to input schema)
            uid: Unique user/session identifier for request tracking and isolation
                 (defaults to 'super-user' for single-user scenarios)
        
        Returns:
            Dictionary containing the application's response data, with any binary
            resources automatically resolved and accessible
        
        Raises:
            Exception: If no connection exists for the app ID or execution fails
        """
        connection = self._connections.get(app_id)
        if not connection:
            raise Exception(f"Connection not found for app ID: {app_id}")

        try:
            # Send request to the remote application
            handler = connection.execute(data, uid)
            result = connection.get_response(handler)

            # Handle resource resolution for binary data (images, models, etc.)
            schema = self.schema(app_id, 'output')
            marshmallow = json_schema_to_marshmallow(schema)
            handle_resources = has_resource_fields(marshmallow())

            # Resolve any resource URLs to actual binary data
            if handle_resources:
                result = resolve_resources("https://" + app_id + "/resource?reid={reid}", result, marshmallow())

            return result
        except Exception as e:
            logging.error(f"[{app_id}] Execution failed: {e}")

    def manifest(self, app_id: str) -> dict:
        """
        Retrieve the manifest metadata for a specific application.
        
        The manifest contains important metadata about the application including
        its capabilities, version information, and operational parameters.
        
        Args:
            app_id: The application ID for which to retrieve the manifest
        
        Returns:
            Dictionary containing the manifest data, or empty dict if not found
        """
        return self._manifest.get(app_id, {})

    def schema(self, app_id: str, type: Literal['input', 'output']) -> dict:
        """
        Retrieve the input or output schema for a specific application.
        
        Schemas define the expected structure and validation rules for data
        sent to (input) or received from (output) the application. They are
        used for validation and automatic resource handling.
        
        Args:
            app_id: The application ID for which to retrieve the schema
            type: The type of schema to retrieve ('input' or 'output')
        
        Returns:
            Dictionary containing the requested schema definition
        
        Raises:
            ValueError: If the schema type is invalid or schema not found
        """
        _input, _output = self._schema.get(app_id, (None, None))

        if type == 'input':
            if _input is None:
                raise ValueError(f"Input schema not found for app ID: {app_id}")
            return _input
        elif type == 'output':
            if _output is None:
                raise ValueError(f"Output schema not found for app ID: {app_id}")
            return _output
        else:
            raise ValueError("Type must be either 'input' or 'output'")
