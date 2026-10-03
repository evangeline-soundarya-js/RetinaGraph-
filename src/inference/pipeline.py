import torch
import torch.nn.functional as F
from src.preprocessing.image_processor import ImageProcessor
from src.features.extractor import FeatureExtractor
from src.graph.builder import GraphBuilder
from src.models.gat_model import RetinaGAT

class InferencePipeline:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor(max_keypoints=150)
        self.graph_builder = GraphBuilder(k_neighbors=5)
        
        # 4 node features: [norm_x, norm_y, intensity, vesselness]
        self.model = RetinaGAT(in_channels=4, hidden_channels=16, num_classes=2, heads=2)
        self.model.eval()
        
        self.model_status = "prototype/untrained"
        
    def run(self, image_bytes: bytes):
        # 1. Preprocessing
        enhanced_image, metadata = self.image_processor.process(image_bytes)
        
        # 2. Feature Extraction
        keypoints, node_features, vessel_mask = self.feature_extractor.extract_features(enhanced_image)
        
        # 3. Graph Construction
        data = self.graph_builder.build_graph(keypoints, node_features)
        
        # 4. GAT Inference
        with torch.no_grad():
            out, attention_weights = self.model(data.x, data.edge_index, batch=None)
            probs = F.softmax(out, dim=1)
            prediction_idx = torch.argmax(probs, dim=1).item()
            
        classes = ["Normal", "Abnormal"]
        
        # 5. Explainability / Evidence
        explanation = {
            "num_nodes": data.num_nodes,
            "num_edges": data.num_edges,
            "important_regions_method": "attention_weights (placeholder)",
            "message": "Model is untrained. This prediction is based on randomly initialized weights."
        }
        
        return {
            "status": "success",
            "prediction": classes[prediction_idx],
            "confidence": None,
            "image_metadata": metadata,
            "graph": {
                "nodes": data.num_nodes,
                "edges": data.num_edges
            },
            "evidence": ["Extracted vessel network graph", "Node connectivity"],
            "explanation": explanation,
            "model_status": self.model_status
        }
