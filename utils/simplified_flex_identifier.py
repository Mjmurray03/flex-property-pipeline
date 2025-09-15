"""
Simplified Flex Property Identifier
Focus: Industrial properties ≥20,000 SF that could be flex/multi-tenant
"""
from typing import Dict, List
import logging

class SimplifiedFlexIdentifier:
    """Identify industrial flex properties using basic criteria"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Target industrial property uses
        self.INDUSTRIAL_USES = [
            'WAREH/DIST TERM',      # Warehouse/Distribution
            'VACANT INDUSTRIAL',    # Vacant Industrial
            'HEAVY MFG',           # Manufacturing
            'LIGHT MFG',           # Light Manufacturing
            'OPEN STORAGE',        # Storage facilities
            'WORKING WATERFRONT',  # Industrial waterfront
            'INDUSTRIAL'           # General industrial
        ]
    
    def identify_flex_candidates(self, min_sqft: int = 20000) -> List[Dict]:
        """
        Find industrial properties that meet flex criteria
        
        Criteria:
        1. Must be industrial property type
        2. Building must be ≥20,000 SF (or multiple buildings totaling ≥20,000 SF)
        3. Suitable for multi-tenant use
        """
        
        # Query for industrial properties only
        pipeline = [
            {
                '$match': {
                    'property_use': {'$in': self.INDUSTRIAL_USES}
                }
            },
            {
                '$project': {
                    'parcel_id': 1,
                    'property_use': 1,
                    'address': 1,
                    'municipality': 1,
                    'owner_name': 1,
                    'acres': 1,
                    'market_value': 1,
                    'assessed_value': 1,
                    'building_data': 1,
                    'flex_indicators': 1
                }
            }
        ]
        
        properties = list(self.db.db.enriched_properties.aggregate(pipeline))
        
        flex_candidates = []
        
        for prop in properties:
            # Check building size
            building_sqft = self._get_building_sqft(prop)
            
            # Skip if building too small or no size data
            if building_sqft < min_sqft and building_sqft > 0:
                continue
            
            # Calculate simple flex score
            flex_score = self._calculate_simple_flex_score(prop, building_sqft)
            
            # Add to candidates if it meets criteria
            if flex_score > 0:  # Any positive score means it's a potential flex
                flex_candidates.append({
                    'parcel_id': prop.get('parcel_id'),
                    'address': prop.get('address', 'Unknown'),
                    'municipality': prop.get('municipality', ''),
                    'property_use': prop.get('property_use'),
                    'building_sqft': building_sqft,
                    'acres': prop.get('acres', 0),
                    'market_value': prop.get('market_value', 0),
                    'owner': prop.get('owner_name', ''),
                    'flex_score': flex_score,
                    'flex_potential': self._determine_flex_potential(flex_score),
                    'multi_tenant_suitable': self._check_multi_tenant_suitability(prop, building_sqft)
                })
        
        # Sort by flex score
        flex_candidates.sort(key=lambda x: x['flex_score'], reverse=True)
        
        self.logger.info(f"Found {len(flex_candidates)} industrial flex candidates")
        
        return flex_candidates
    
    def _get_building_sqft(self, property_data: Dict) -> int:
        """Extract building square footage"""
        
        # Try different fields where sqft might be stored
        if property_data.get('building_data'):
            building = property_data['building_data']
            
            # Check various sqft fields
            sqft = building.get('total_area', 0) or building.get('total_sqft', 0)
            
            # If area is given in acres, convert (1 acre = 43,560 sqft)
            if sqft == 0 and building.get('area'):
                area = building['area']
                if area < 100:  # Likely acres if small number
                    sqft = area * 43560
            
            return int(sqft)
        
        # Estimate from acres if no building data (assume 25% coverage)
        acres = property_data.get('acres', 0)
        if acres > 0:
            # Rough estimate: 25% lot coverage
            estimated_sqft = acres * 43560 * 0.25
            return int(estimated_sqft)
        
        return 0
    
    def _calculate_simple_flex_score(self, prop: Dict, building_sqft: int) -> float:
        """
        Simple scoring focused on flex suitability
        
        Scoring factors:
        - Property type (warehouse/distribution = best)
        - Building size (20K-100K SF = ideal)
        - Lot size (1-10 acres = ideal)
        - Market value (indicates investment potential)
        """
        
        score = 0
        
        # 1. Property Use Type (0-4 points)
        use_type = prop.get('property_use', '').upper()
        if 'WAREH' in use_type or 'DIST' in use_type:
            score += 4  # Perfect for flex
        elif 'VACANT INDUSTRIAL' in use_type:
            score += 3  # Good potential
        elif 'MFG' in use_type:
            score += 2  # Can be converted
        elif 'STORAGE' in use_type:
            score += 2  # Often multi-tenant
        elif 'INDUSTRIAL' in use_type:
            score += 1  # Generic industrial
        
        # 2. Building Size (0-3 points)
        if 20000 <= building_sqft <= 50000:
            score += 3  # Perfect flex size
        elif 50000 < building_sqft <= 100000:
            score += 2  # Good for multi-tenant
        elif 100000 < building_sqft <= 200000:
            score += 1  # Large but subdividable
        elif building_sqft > 200000:
            score += 0.5  # Very large, harder to subdivide
        
        # 3. Lot Size (0-2 points)
        acres = prop.get('acres', 0)
        if 1 <= acres <= 5:
            score += 2  # Ideal flex lot size
        elif 5 < acres <= 10:
            score += 1.5  # Still good
        elif 0.5 <= acres < 1:
            score += 1  # Small but workable
        elif 10 < acres <= 20:
            score += 0.5  # Large but manageable
        
        # 4. Market Value Range (0-2 points)
        market_value = prop.get('market_value', 0)
        if 500000 <= market_value <= 5000000:
            score += 2  # Typical flex value range
        elif 5000000 < market_value <= 10000000:
            score += 1  # Higher end flex
        elif 250000 <= market_value < 500000:
            score += 1  # Lower end but viable
        
        # 5. Bonus for known flex indicators
        if prop.get('flex_indicators'):
            indicators = prop['flex_indicators']
            if indicators.get('is_flex_compatible'):
                score += 1
            if indicators.get('property_type_score', 0) > 2:
                score += 0.5
        
        return round(score, 2)
    
    def _determine_flex_potential(self, score: float) -> str:
        """Categorize flex potential"""
        if score >= 8:
            return "EXCELLENT"
        elif score >= 6:
            return "VERY GOOD"
        elif score >= 4:
            return "GOOD"
        elif score >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _check_multi_tenant_suitability(self, prop: Dict, building_sqft: int) -> str:
        """Determine multi-tenant suitability"""
        
        # Building size suitability
        if 20000 <= building_sqft <= 100000:
            size_suitable = True
        else:
            size_suitable = False
        
        # Property type suitability
        use_type = prop.get('property_use', '').upper()
        type_suitable = any(term in use_type for term in ['WAREH', 'DIST', 'STORAGE', 'VACANT'])
        
        # Acreage suitability (parking for multiple tenants)
        acres = prop.get('acres', 0)
        acres_suitable = 1 <= acres <= 10
        
        if size_suitable and type_suitable and acres_suitable:
            return "HIGHLY SUITABLE"
        elif size_suitable and type_suitable:
            return "SUITABLE"
        elif size_suitable or type_suitable:
            return "POTENTIALLY SUITABLE"
        else:
            return "LIMITED SUITABILITY"
    
    def generate_report(self, candidates: List[Dict]) -> Dict:
        """Generate summary report of flex candidates"""
        
        if not candidates:
            return {'error': 'No candidates found'}
        
        report = {
            'total_candidates': len(candidates),
            'by_potential': {},
            'by_municipality': {},
            'size_distribution': {
                '20-50K SF': 0,
                '50-100K SF': 0,
                '100-200K SF': 0,
                '200K+ SF': 0,
                'Unknown Size': 0
            },
            'top_10': candidates[:10],
            'multi_tenant_suitable': 0
        }
        
        # Analyze candidates
        for candidate in candidates:
            # By potential
            potential = candidate['flex_potential']
            if potential not in report['by_potential']:
                report['by_potential'][potential] = 0
            report['by_potential'][potential] += 1
            
            # By municipality
            muni = candidate.get('municipality', 'Unknown')
            if muni not in report['by_municipality']:
                report['by_municipality'][muni] = 0
            report['by_municipality'][muni] += 1
            
            # Size distribution
            sqft = candidate.get('building_sqft', 0)
            if sqft == 0:
                report['size_distribution']['Unknown Size'] += 1
            elif sqft <= 50000:
                report['size_distribution']['20-50K SF'] += 1
            elif sqft <= 100000:
                report['size_distribution']['50-100K SF'] += 1
            elif sqft <= 200000:
                report['size_distribution']['100-200K SF'] += 1
            else:
                report['size_distribution']['200K+ SF'] += 1
            
            # Multi-tenant suitability
            if 'SUITABLE' in candidate.get('multi_tenant_suitable', ''):
                report['multi_tenant_suitable'] += 1
        
        return report

# Usage
def find_true_flex_properties(db_manager):
    """Find industrial properties suitable for flex/multi-tenant use"""
    
    identifier = SimplifiedFlexIdentifier(db_manager)
    
    # Find candidates
    candidates = identifier.identify_flex_candidates(min_sqft=20000)
    
    # Generate report
    report = identifier.generate_report(candidates)
    
    print("\n" + "="*50)
    print("INDUSTRIAL FLEX PROPERTY ANALYSIS")
    print("="*50)
    
    print(f"\nTotal Industrial Flex Candidates: {report['total_candidates']}")
    
    print("\nBy Flex Potential:")
    for potential, count in report['by_potential'].items():
        print(f"  {potential}: {count}")
    
    print("\nBy Municipality:")
    for muni, count in sorted(report['by_municipality'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {muni}: {count}")
    
    print("\nSize Distribution:")
    for size_range, count in report['size_distribution'].items():
        if count > 0:
            print(f"  {size_range}: {count}")
    
    print(f"\nMulti-Tenant Suitable: {report['multi_tenant_suitable']}")
    
    print("\nTop 10 Flex Candidates:")
    for i, candidate in enumerate(report['top_10'], 1):
        print(f"\n{i}. {candidate['address']}")
        print(f"   Municipality: {candidate['municipality']}")
        print(f"   Property Use: {candidate['property_use']}")
        print(f"   Building Size: {candidate['building_sqft']:,} SF")
        print(f"   Acres: {candidate['acres']:.2f}")
        print(f"   Market Value: ${candidate['market_value']:,.0f}")
        print(f"   Flex Score: {candidate['flex_score']}")
        print(f"   Potential: {candidate['flex_potential']}")
        print(f"   Multi-Tenant: {candidate['multi_tenant_suitable']}")
    
    return candidates

if __name__ == "__main__":
    print("Run after enrichment to identify true flex properties")