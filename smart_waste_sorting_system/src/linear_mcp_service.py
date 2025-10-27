"""
Linear MCP Integration Service for Smart Waste Sorting System
This service integrates with Linear MCP to manage industry data and waste processing workflows
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class IndustryApplication:
    """Data class for industry application information"""
    waste_type: str
    industry_type: str
    applications: str
    recycling_process: str
    market_value: float
    environmental_impact: str
    linear_issue_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class WasteDetection:
    """Data class for waste detection results"""
    waste_type: str
    confidence: float
    category: str
    timestamp: str
    location: Optional[str] = None
    linear_issue_id: Optional[str] = None

class LinearMCPIntegration:
    """Integration service for Linear MCP server"""
    
    def __init__(self):
        self.industry_applications = self._load_industry_data()
        self.detection_history = []
        
    def _load_industry_data(self) -> Dict[str, IndustryApplication]:
        """Load industry application data"""
        return {
            'Banana_Peel': IndustryApplication(
                waste_type='Banana_Peel',
                industry_type='Composting & Agriculture',
                applications='Used in composting facilities, biogas production, and organic fertilizer manufacturing. High nutrient content makes it valuable for soil enrichment.',
                recycling_process='Collection → Sorting → Composting → Quality Control → Distribution',
                market_value=0.15,  # USD per kg
                environmental_impact='Reduces methane emissions, improves soil health, reduces chemical fertilizer use',
                created_at=datetime.now().isoformat()
            ),
            'Orange_Peel': IndustryApplication(
                waste_type='Orange_Peel',
                industry_type='Food Processing & Cosmetics',
                applications='Extracted for essential oils, pectin production, and natural flavoring. Used in cosmetics, cleaning products, and food additives.',
                recycling_process='Collection → Cleaning → Oil Extraction → Processing → Packaging',
                market_value=2.50,  # USD per kg
                environmental_impact='Reduces waste, creates valuable byproducts, supports circular economy',
                created_at=datetime.now().isoformat()
            ),
            'Plastic': IndustryApplication(
                waste_type='Plastic',
                industry_type='Recycling & Manufacturing',
                applications='Processed into new plastic products, textile fibers, construction materials, and packaging. Various plastic types have different recycling applications.',
                recycling_process='Collection → Sorting → Cleaning → Shredding → Melting → Molding',
                market_value=0.80,  # USD per kg
                environmental_impact='Reduces plastic pollution, conserves petroleum resources, reduces energy consumption',
                created_at=datetime.now().isoformat()
            ),
            'Paper': IndustryApplication(
                waste_type='Paper',
                industry_type='Pulp & Paper Industry',
                applications='Recycled into new paper products, cardboard, insulation materials, and packaging. Reduces deforestation and energy consumption.',
                recycling_process='Collection → Sorting → Pulping → Cleaning → Drying → Rolling',
                market_value=0.25,  # USD per kg
                environmental_impact='Saves trees, reduces water usage, decreases landfill waste',
                created_at=datetime.now().isoformat()
            ),
            'Wood': IndustryApplication(
                waste_type='Wood',
                industry_type='Construction & Furniture',
                applications='Used for particle board, mulch, biomass fuel, and construction materials. Can be processed into engineered wood products.',
                recycling_process='Collection → Sorting → Chipping → Processing → Quality Control → Distribution',
                market_value=0.30,  # USD per kg
                environmental_impact='Reduces deforestation, creates renewable energy, supports sustainable construction',
                created_at=datetime.now().isoformat()
            )
        }
    
    def get_industry_applications(self, waste_types: List[str]) -> List[IndustryApplication]:
        """Get industry applications for specific waste types"""
        applications = []
        
        for waste_type in waste_types:
            if waste_type in self.industry_applications:
                applications.append(self.industry_applications[waste_type])
            else:
                logger.warning(f"Unknown waste type: {waste_type}")
        
        return applications
    
    def create_linear_issue(self, detection: WasteDetection) -> Optional[str]:
        """
        Create a Linear issue for waste detection tracking
        This would integrate with the actual Linear MCP server
        """
        try:
            # In a real implementation, this would call the Linear MCP server
            # For now, we'll simulate the issue creation
            
            issue_data = {
                'title': f'Waste Detection: {detection.waste_type}',
                'description': f'''
**Waste Detection Alert**

- **Waste Type**: {detection.waste_type}
- **Category**: {detection.category}
- **Confidence**: {detection.confidence:.2%}
- **Timestamp**: {detection.timestamp}
- **Location**: {detection.location or 'Unknown'}

**Industry Applications**:
{self._get_industry_summary(detection.waste_type)}

**Action Required**: Process waste according to industry guidelines
                ''',
                'labels': ['waste-detection', detection.category, 'automated'],
                'priority': self._get_priority(detection.confidence),
                'assignee': 'waste-processing-team'
            }
            
            # Simulate Linear issue creation
            issue_id = f"WASTE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            logger.info(f"Created Linear issue: {issue_id}")
            
            return issue_id
            
        except Exception as e:
            logger.error(f"Error creating Linear issue: {e}")
            return None
    
    def _get_industry_summary(self, waste_type: str) -> str:
        """Get industry summary for waste type"""
        if waste_type in self.industry_applications:
            app = self.industry_applications[waste_type]
            return f"- Industry: {app.industry_type}\n- Applications: {app.applications}\n- Market Value: ${app.market_value}/kg"
        return "No industry information available"
    
    def _get_priority(self, confidence: float) -> str:
        """Determine priority based on detection confidence"""
        if confidence >= 0.9:
            return 'high'
        elif confidence >= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def track_detection(self, detection: WasteDetection) -> str:
        """Track waste detection and create Linear issue"""
        # Add to detection history
        self.detection_history.append(detection)
        
        # Create Linear issue
        issue_id = self.create_linear_issue(detection)
        detection.linear_issue_id = issue_id
        
        logger.info(f"Tracked detection: {detection.waste_type} (Issue: {issue_id})")
        return issue_id
    
    def get_detection_stats(self) -> Dict:
        """Get detection statistics"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'organic_count': 0,
                'inorganic_count': 0,
                'recycling_rate': 0,
                'linear_issues_created': 0
            }
        
        total = len(self.detection_history)
        organic = sum(1 for d in self.detection_history if d.category == 'organic')
        inorganic = sum(1 for d in self.detection_history if d.category == 'inorganic')
        issues_created = sum(1 for d in self.detection_history if d.linear_issue_id)
        
        return {
            'total_detections': total,
            'organic_count': organic,
            'inorganic_count': inorganic,
            'recycling_rate': (inorganic / total * 100) if total > 0 else 0,
            'linear_issues_created': issues_created
        }
    
    def export_data(self) -> Dict:
        """Export all data for backup or analysis"""
        return {
            'industry_applications': {k: asdict(v) for k, v in self.industry_applications.items()},
            'detection_history': [asdict(d) for d in self.detection_history],
            'stats': self.get_detection_stats(),
            'export_timestamp': datetime.now().isoformat()
        }

# Global instance
linear_mcp_service = LinearMCPIntegration()

def get_linear_mcp_service() -> LinearMCPIntegration:
    """Get the global Linear MCP service instance"""
    return linear_mcp_service
