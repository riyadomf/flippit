// src/MapComponent.js

import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css'; // Import leaflet's CSS

// --- FIX for a known issue with react-leaflet and webpack ---
// This ensures that the default marker icons are loaded correctly.
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow
});
L.Marker.prototype.options.icon = DefaultIcon;
// --- End of FIX ---

const MapComponent = ({ properties }) => {
    // My Thought Process: We need a default center for the map.
    // Hardcoding the coordinates for Warren, MI is the simplest and most reliable
    // way to ensure the map is always centered correctly on startup.
    const warrenCoords = [42.4928, -83.0280]; // Center of Warren, MI

    return (
        <div className="map-container">
            <MapContainer center={warrenCoords} zoom={12} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                />

                {/* Loop through all properties and create a marker for each */}
                {properties.map(prop => (
                    // Ensure the property has valid coordinates before rendering a marker
                    prop.latitude && prop.longitude ? (
                        <Marker key={prop.property_id} position={[prop.latitude, prop.longitude]}>
                            <Popup>
                                <strong>{prop.address}</strong><br />
                                <span>List Price: ${prop.list_price.toLocaleString()}</span><br />
                                <span>Est. ROI: {prop.roi_percentage.toFixed(1)}%</span><br />
                                <span>Grade: {prop.overall_grade}</span>
                            </Popup>
                        </Marker>
                    ) : null
                ))}
            </MapContainer>
        </div>
    );
};

export default MapComponent;