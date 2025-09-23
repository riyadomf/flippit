// src/App.js

import React, { useState, useEffect, useCallback, useRef } from 'react';
import './App.css';
import PropertyModal from './PropertyModal';
import MapComponent from './MapComponent';

// API base URL - ensure your FastAPI backend is running on this port
const API_URL = "http://127.0.0.1:8000";

// --- Child Components ---

const PropertyCard = ({ property, onCardClick }) => {
    const getGradeClass = (grade) => `grade grade-${grade}`;
    const formatCurrency = (value) => value.toLocaleString('en-US', { style: 'currency', currency: 'USD' });

    return (
        // --- 3. ADD THE onClick HANDLER ---
        <div className="property-card" onClick={() => onCardClick(property)}>
            {property.primary_photo && (
                <img 
                    src={property.primary_photo} 
                    alt={`Exterior of ${property.address}`} 
                    className="card-image" 
                />
            )}
            <h3>{property.address}</h3>
            <div className="card-info">
                List Price: <span>{formatCurrency(property.list_price)}</span>
            </div>
            <div className="card-info">
                Est. Resale: <span>{formatCurrency(property.estimated_resale_price)}</span>
            </div>
            <div className="card-info">
                Estimated ROI: <span>{property.roi_percentage.toFixed(1)}%</span>
            </div>
            <div className={getGradeClass(property.overall_grade)}>
                {property.overall_grade}
            </div>
        </div>
    );
};

// --- Main App Component ---

function App() {
    const [properties, setProperties] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isScoring, setIsScoring] = useState(false);
    const [filters, setFilters] = useState({ min_roi: '', max_price: '' });
    const [error, setError] = useState(null);
    const [selectedProperty, setSelectedProperty] = useState(null);
    const [view, setView] = useState('list'); // 'list' or 'map'
    // useRef is perfect for storing interval IDs without causing re-renders.
    const pollingInterval = useRef(null);

    // My Thought Process: A single, reusable function to fetch properties.
    // It builds the query string from the current filters state.
    // useCallback prevents this function from being recreated on every render,
    // which is a good practice for functions used in useEffect.
    const fetchProperties = useCallback(async () => {
        // ... (this function remains exactly the same)
        setIsLoading(true);
        setError(null);
        const params = new URLSearchParams();
        if (filters.min_roi) params.append('min_roi', filters.min_roi);
        if (filters.max_price) params.append('max_price', filters.max_price);
        try {
            const response = await fetch(`${API_URL}/properties?${params.toString()}`);
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            const data = await response.json();
            setProperties(data);
        } catch (e) {
            console.error("Failed to fetch properties:", e);
            setError("Could not load property data. Is the backend server running?");
        } finally {
            setIsLoading(false);
        }
    }, [filters]);

    // This useEffect is for cleanup. If the component unmounts, we must stop polling.
    useEffect(() => {
        return () => {
            if (pollingInterval.current) {
                clearInterval(pollingInterval.current);
            }
        };
    }, []);

    const handleScoreClick = async () => {
        setIsScoring(true);
        setError(null);

        // --- NEW: Polling Logic ---
        const checkStatus = async () => {
            try {
                const statusResponse = await fetch(`${API_URL}/scoring-status`);
                const statusData = await statusResponse.json();

                if (!statusData.is_scoring) {
                    console.log("Scoring complete. Refreshing properties.");
                    clearInterval(pollingInterval.current);
                    setIsScoring(false);
                    fetchProperties(); // Refresh the list now that scoring is done
                } else {
                    console.log("Still scoring...");
                }
            } catch (e) {
                console.error("Polling error:", e);
                setError("Error checking scoring status.");
                clearInterval(pollingInterval.current);
                setIsScoring(false);
            }
        };

        try {
            const response = await fetch(`${API_URL}/process-scores`, { method: 'POST' });
            if (response.status === 409) { // Conflict
                alert("A scoring process is already running. Please wait.");
                setIsScoring(false);
                return;
            }
            if (response.status !== 202) {
                throw new Error("Failed to start scoring process.");
            }
            // If the request was accepted, start polling.
            pollingInterval.current = setInterval(checkStatus, 2000); // Check every 2 seconds

        } catch (e) {
            console.error("Scoring trigger failed:", e);
            setError("Could not start the scoring process.");
            setIsScoring(false);
        }
    };

    const handleFilterChange = (e) => {
        const { name, value } = e.target;
        setFilters(prev => ({ ...prev, [name]: value }));
    };

    
    // This useEffect handles re-fetching data when filters are applied by the user.
    // It's separate from the initial load effect.
    useEffect(() => {
        const handler = setTimeout(() => {
            fetchProperties();
        }, 500); // Debounce: Wait 500ms after user stops typing to fetch

        return () => {
            clearTimeout(handler); // Cleanup on unmount or if filters change again
        };
    }, [filters, fetchProperties]);

    const handleCardClick = (property) => {
        setSelectedProperty(property);
    };

    const handleCloseModal = () => {
        setSelectedProperty(null);
    };

    return (
        <div className="App">
            <header>
                <h1>Flippit Deal Finder</h1>
                <p>Find potential real estate investment deals in Warren.</p>
            </header>


            {/* --- 3. ADD THE VIEW TOGGLE BUTTONS --- */}
            <div className="view-toggle">
                <button className={view === 'list' ? 'active' : ''} onClick={() => setView('list')}>
                    List View
                </button>
                <button className={view === 'map' ? 'active' : ''} onClick={() => setView('map')}>
                    Map View
                </button>
            </div>


            <div className="controls-container">
                <div className="filter-group">
                    <label htmlFor="min_roi">Min. ROI (%)</label>
                    <input type="number" id="min_roi" name="min_roi" value={filters.min_roi} onChange={handleFilterChange} placeholder="e.g., 15" />
                </div>
                <div className="filter-group">
                    <label htmlFor="max_price">Max. List Price ($)</label>
                    <input type="number" id="max_price" name="max_price" value={filters.max_price} onChange={handleFilterChange} placeholder="e.g., 200000" />
                </div>
                <button className="score-button" onClick={handleScoreClick} disabled={isScoring}>
                    {isScoring ? 'Scoring in Progress...' : 'Score New Properties'}
                </button>
            </div>

            {/* Display loader if fetching OR if scoring is in progress */}
            {(isLoading || isScoring) && (
                <div className="loader">
                    <div className="spinner"></div> {/* <-- THIS IS THE NEW LINE */}
                    <p>{isScoring ? 'Processing Scores...' : 'Loading Properties...'}</p>
                </div>
            )}
            
            {error && <div className="loader" style={{color: 'red'}}>{error}</div>}
            
            {!isLoading && !isScoring && !error && (
                <div>
                    {/* --- CONDITIONAL RENDERING FOR LIST OR MAP --- */}
                    {view === 'list' ? (
                        <div className="property-list">
                            {properties.length > 0 ? (
                                properties.map(prop => (
                                    <PropertyCard key={prop.property_id} property={prop} onCardClick={handleCardClick} />
                                ))
                            ) : (
                                <p>No properties found matching your criteria.</p>
                            )}
                        </div>
                    ) : (
                        <MapComponent properties={properties} />
                    )}
                </div>
            )}

            {/* ---  CONDITIONALLY RENDER THE MODAL --- */}
            {selectedProperty && (
                <PropertyModal property={selectedProperty} onClose={handleCloseModal} />
            )}
        </div>
    );
}

export default App;