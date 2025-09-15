"""
Flex Property Scorer
Calculates flex industrial indicators and scores
"""
from typing import Dict, Tuple, List, Optional
from datetime import datetime
import logging

class FlexPropertyScorer:
    """Score properties for flex industrial potential"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights
        self.weights = {
            'zoning': 3.0,           # Most important
            'size': 2.0,             # Property size
            'building': 2.5,         # Building characteristics  
            'location': 1.5,         # Location factors
            'activity': 1.0,         # Recent activity
            'value': 1.0             # Value indicators
        }
        
        # Ideal flex property characteristics
        self.ideal_flex = {
            'min_acres': 0.5,
            'max_acres': 20,
            'ideal_acres': 2.5,
            'min_building_sqft': 5000,
            'max_building_sqft': 100000,
            'ideal_building_sqft': 25000,
            'min_ceiling_height': 14,
            'max_ceiling_height': 24,
            'ideal_office_ratio': 0.25,  # 25% office
            'min_loading_doors': 1,
            'ideal_loading_doors': 4
        }
    
    def calculate_flex_score(self, property_data: Dict) -> Tuple[float, Dict]:
        """
        Calculate flex score for a property
        
        Returns:
            Tuple of (score, indicators_dict)
        """
        
        indicators = {
            'zoning_score': 0,
            'size_score': 0,
            'building_score': 0,
            'location_score': 0,
            'activity_score': 0,
            'value_score': 0,
            'flags': []
        }
        
        # 1. Zoning Score (0-10)
        indicators['zoning_score'] = self._score_zoning(property_data)
        
        # 2. Size Score (0-10)
        indicators['size_score'] = self._score_size(property_data)
        
        # 3. Building Score (0-10)
        indicators['building_score'] = self._score_building(property_data)
        
        # 4. Location Score (0-10)
        indicators['location_score'] = self._score_location(property_data)
        
        # 5. Activity Score (0-10)
        indicators['activity_score'] = self._score_activity(property_data)
        
        # 6. Value Score (0-10)
        indicators['value_score'] = self._score_value(property_data)
        
        # Calculate weighted total score
        total_score = 0
        total_weight = 0
        
        score_components = {
            'zoning_score': self.weights['zoning'],
            'size_score': self.weights['size'],
            'building_score': self.weights['building'],
            'location_score': self.weights['location'],
            'activity_score': self.weights['activity'],
            'value_score': self.weights['value']
        }
        
        for component, weight in score_components.items():
            score = indicators.get(component, 0)
            total_score += score * weight
            total_weight += weight
        
        # Normalize to 0-10 scale
        final_score = (total_score / total_weight) if total_weight > 0 else 0
        
        # Add bonus flags
        if indicators['zoning_score'] >= 8 and indicators['size_score'] >= 7:
            indicators['flags'].append('HIGH_POTENTIAL')
            final_score += 0.5
        
        if indicators['building_score'] >= 8:
            indicators['flags'].append('IDEAL_BUILDING')
            final_score += 0.3
        
        # Cap at 10
        final_score = min(10, final_score)
        
        return round(final_score, 2), indicators
    
    def _score_zoning(self, property_data: Dict) -> float:
        """Score based on zoning classification"""
        
        zoning = property_data.get('zoning', '').upper()
        
        # Perfect flex zones
        perfect_zones = ['IL', 'IP', 'PIPD', 'M-1']
        if zoning in perfect_zones:
            return 10
        
        # Good industrial zones
        good_zones = ['IG', 'IND', 'M-2']
        if zoning in good_zones:
            return 8
        
        # Mixed use with industrial
        mixed_zones = ['MUPD', 'MXPD', 'AGR/IND']
        if zoning in mixed_zones:
            return 6
        
        # Commercial zones that might work
        commercial_zones = ['CG', 'CH', 'CS']
        if zoning in commercial_zones:
            return 4
        
        return 0
    
    def _score_size(self, property_data: Dict) -> float:
        """Score based on property size"""
        
        acres = property_data.get('acres', 0) or 0  # Ensure not None
        
        if acres <= 0:
            return 0
        
        # Too small
        if acres < self.ideal_flex['min_acres']:
            return 2
        
        # Too large (unless it's subdividable)
        if acres > self.ideal_flex['max_acres']:
            return 3
        
        # Calculate score based on proximity to ideal
        if self.ideal_flex['min_acres'] <= acres <= self.ideal_flex['max_acres']:
            # Peak score at ideal size
            ideal = self.ideal_flex['ideal_acres']
            
            if acres == ideal:
                return 10
            elif acres < ideal:
                # Scale from min to ideal
                range_size = ideal - self.ideal_flex['min_acres']
                position = acres - self.ideal_flex['min_acres']
                return 5 + (position / range_size) * 5
            else:
                # Scale from ideal to max
                range_size = self.ideal_flex['max_acres'] - ideal
                position = self.ideal_flex['max_acres'] - acres
                return 5 + (position / range_size) * 5
        
        return 5
    
    def _score_building(self, property_data: Dict) -> float:
        """Score based on building characteristics"""
        
        score = 5  # Base score if building exists
        
        building_data = property_data.get('building_data', {})
        if not building_data:
            # Check if there's any improvement value
            if property_data.get('improvement_value', 0) > 100000:
                return 3  # Some building exists
            return 0
        
        # Building size
        building_sqft = building_data.get('total_sqft', 0)
        if self.ideal_flex['min_building_sqft'] <= building_sqft <= self.ideal_flex['max_building_sqft']:
            score += 2
            
            # Bonus for ideal size
            if 15000 <= building_sqft <= 35000:
                score += 1
        
        # Ceiling height (critical for flex)
        height = building_data.get('ceiling_height', 0)
        if height > 0:
            if self.ideal_flex['min_ceiling_height'] <= height <= self.ideal_flex['max_ceiling_height']:
                score += 2
        
        # Loading capabilities
        loading_doors = building_data.get('loading_docks', 0) + building_data.get('overhead_doors', 0)
        if loading_doors >= self.ideal_flex['min_loading_doors']:
            score += 1
        
        # Office/warehouse mix
        office_area = building_data.get('office_area', 0)
        warehouse_area = building_data.get('warehouse_area', 0)
        
        if office_area > 0 and warehouse_area > 0:
            ratio = office_area / (office_area + warehouse_area)
            if 0.15 <= ratio <= 0.50:  # 15-50% office is ideal for flex
                score += 2
        
        return min(10, score)
    
    def _score_location(self, property_data: Dict) -> float:
        """Score based on location factors"""
        
        score = 5  # Base score
        
        # Municipality scoring (some areas are better for flex)
        municipality = property_data.get('municipality', '').upper()
        
        prime_locations = ['BOCA RATON', 'DELRAY BEACH', 'BOYNTON BEACH', 'WEST PALM BEACH']
        good_locations = ['LAKE WORTH', 'RIVIERA BEACH', 'JUPITER', 'WELLINGTON']
        
        if any(loc in municipality for loc in prime_locations):
            score += 3
        elif any(loc in municipality for loc in good_locations):
            score += 2
        
        # Near highways (would need GIS analysis)
        # For now, use a placeholder
        if property_data.get('near_highway', False):
            score += 2
        
        return min(10, score)
    
    def _score_activity(self, property_data: Dict) -> float:
        """Score based on recent activity"""
        
        score = 5  # Base score
        
        # Recent permits
        permits = property_data.get('permits', [])
        recent_permits = [p for p in permits if self._is_recent(p.get('issue_date'))]
        
        if recent_permits:
            score += min(3, len(recent_permits))
        
        # Active business licenses
        licenses = property_data.get('business_licenses', [])
        active_licenses = [l for l in licenses if l.get('status') == 'ACTIVE']
        
        if active_licenses:
            # Multiple tenants suggests flex use
            if 2 <= len(active_licenses) <= 10:
                score += 3
            elif len(active_licenses) == 1:
                score += 1
        
        # Recent sale (suggests market activity)
        sale_date = property_data.get('sale_date')
        if sale_date and self._is_recent(sale_date, days=730):  # Within 2 years
            score += 1
        
        return min(10, score)
    
    def _score_value(self, property_data: Dict) -> float:
        """Score based on value indicators"""
        
        score = 5  # Base score
        
        improvement_value = property_data.get('improvement_value', 0)
        land_value = property_data.get('land_market_value', 1)
        
        if land_value > 0:
            improvement_ratio = improvement_value / land_value
            
            # Good improvement to land ratio (building exists and is valuable)
            if improvement_ratio > 1.5:
                score += 2
            elif improvement_ratio > 0.5:
                score += 1
        
        # Absolute improvement value
        if 500000 <= improvement_value <= 5000000:
            score += 2
        elif 250000 <= improvement_value < 500000:
            score += 1
        
        # Price per square foot (if available)
        building_sqft = property_data.get('building_data', {}).get('total_sqft', 0)
        if building_sqft > 0 and improvement_value > 0:
            price_per_sqft = improvement_value / building_sqft
            
            # Reasonable price range for flex
            if 50 <= price_per_sqft <= 150:
                score += 1
        
        return min(10, score)
    
    def _is_recent(self, date_str: Optional[str], days: int = 365) -> bool:
        """Check if a date is within the specified number of days"""
        
        if not date_str:
            return False
        
        try:
            if isinstance(date_str, str):
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date = date_str
            
            days_ago = (datetime.utcnow() - date).days
            return days_ago <= days
        except:
            return False
    
    def get_flex_classification(self, score: float) -> str:
        """Classify property based on flex score"""
        
        if score >= 8:
            return "PRIME_FLEX"
        elif score >= 6:
            return "GOOD_FLEX"
        elif score >= 4:
            return "POTENTIAL_FLEX"
        elif score >= 2:
            return "POSSIBLE_FLEX"
        else:
            return "UNLIKELY_FLEX"
    
    def generate_report(self, property_data: Dict, score: float, 
                       indicators: Dict) -> Dict:
        """Generate detailed flex analysis report"""
        
        classification = self.get_flex_classification(score)
        
        report = {
            'parcel_id': property_data.get('parcel_id'),
            'address': property_data.get('site_address', 'Unknown'),
            'flex_score': score,
            'classification': classification,
            'indicators': indicators,
            'strengths': [],
            'weaknesses': [],
            'recommendations': []
        }
        
        # Identify strengths
        if indicators['zoning_score'] >= 8:
            report['strengths'].append('Excellent industrial zoning')
        if indicators['size_score'] >= 8:
            report['strengths'].append('Ideal property size for flex')
        if indicators['building_score'] >= 8:
            report['strengths'].append('Building well-suited for flex use')
        
        # Identify weaknesses
        if indicators['zoning_score'] < 5:
            report['weaknesses'].append('Zoning may not permit industrial use')
        if indicators['size_score'] < 5:
            report['weaknesses'].append('Property size not optimal for flex')
        if indicators['building_score'] < 5:
            report['weaknesses'].append('Building needs modifications for flex use')
        
        # Recommendations
        if classification in ['PRIME_FLEX', 'GOOD_FLEX']:
            report['recommendations'].append('High priority for acquisition/development')
        elif classification == 'POTENTIAL_FLEX':
            report['recommendations'].append('Investigate further with site visit')
            report['recommendations'].append('Check zoning for conditional use permits')
        
        return report
