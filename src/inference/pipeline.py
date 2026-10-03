import torch
import torch.nn.functional as F
from src.preprocessing.image_processor import ImageProcessor
from src.features.extractor import FeatureExtractor
from src.graph.builder import GraphBuilder
from src.models.gat_model import RetinaGAT
from src.preprocessing.fundus_validator import FundusValidator
import io
from PIL import Image
import numpy as np

class InferencePipeline:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.feature_extractor = FeatureExtractor(max_keypoints=150)
        self.graph_builder = GraphBuilder(k_neighbors=5)
        self.validator = FundusValidator()
        
        # 4 node features: [norm_x, norm_y, intensity, vesselness]
        self.model = RetinaGAT(in_channels=4, hidden_channels=16, num_classes=2, heads=2)
        self.model.eval()
        
        self.model_status = "prototype/untrained"
        
    def run(self, image_bytes: bytes):
        # 0. Graceful Decoding & Validation
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img_np = np.array(img)
            # Ensure it has 3 channels for validation if it's grayscale
            if len(img_np.shape) == 2:
                import cv2
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            elif img_np.shape[2] == 4:
                import cv2
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        except Exception as e:
            return {
                "status": "rejected",
                "reason": "Image is corrupted or in an unsupported format.",
                "checks": {"decodable": False},
                "prediction": None,
                "confidence": None,
                "image_metadata": {},
                "graph": None,
                "evidence": [],
                "explanation": None,
                "model_status": "not_run"
            }
            
        validation_result = self.validator.validate(img_np)
        if validation_result["status"] == "rejected":
            return {
                "status": "rejected",
                "reason": validation_result["reason"],
                "checks": validation_result["checks"],
                "prediction": None,
                "confidence": None,
                "image_metadata": {
                    "original_size": img.size,
                    "format": img.format,
                    "mode": img.mode
                },
                "graph": None,
                "evidence": [],
                "explanation": None,
                "model_status": "not_run"
            }

        # 1. Preprocessing
        prep_result = self.image_processor.process(image_bytes)
        metadata = prep_result["metadata"]
        enhanced_image = prep_result["vessel_representation"]
        
        # 2. Feature Extraction
        feature_result = self.feature_extractor.extract_features(enhanced_image)
        
        if feature_result["status"] == "insufficient_retinal_structure":
            return {
                "status": "insufficient_retinal_structure",
                "reason": "No valid retinal structure (vessels, junctions) could be detected in the image.",
                "checks": {"features_found": False},
                "prediction": None,
                "confidence": None,
                "image_metadata": metadata,
                "graph": None,
                "evidence": [],
                "explanation": None,
                "model_status": "not_run"
            }
            
        keypoints = feature_result["keypoints"]
        node_features = feature_result["node_features"]
        vessel_mask = feature_result["vessel_mask"]
        
        # 3. Graph Construction
        data = self.graph_builder.build_graph(keypoints, node_features)
        
        # 4. GAT Inference
        # Validate Dimensions Explicitly
        if data.x.shape[1] != self.model.config["in_channels"]:
            raise ValueError(f"Expected node feature dimension {self.model.config['in_channels']}, received {data.x.shape[1]}")
            
        if data.edge_attr.shape[1] != self.model.config["edge_dim"]:
            raise ValueError(f"Expected edge attribute dimension {self.model.config['edge_dim']}, received {data.edge_attr.shape[1]}")
            
        with torch.no_grad():
            out, attention_weights = self.model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=None)
            probs = F.softmax(out, dim=1)
            raw_logits = out.cpu().numpy().tolist()
            
        # 5. Explainability / Evidence
        explanation = {
            "num_nodes": data.stats["num_nodes"],
            "num_edges": data.stats["num_edges"],
            "important_regions_method": "attention_weights (placeholder)",
            "message": "Model is untrained. This prediction is based on randomly initialized weights."
        }
        
        return {
            "status": "success",
            "prediction": None,
            "confidence": None,
            "raw_logits": raw_logits,
            "image_metadata": metadata,
            "graph": {
                "nodes": data.stats["num_nodes"],
                "edges": data.stats["num_edges"],
                "avg_degree": data.stats["avg_degree"],
                "connected_components": data.stats["connected_components"],
                "density": data.stats["density"],
                "isolated_nodes": data.stats["isolated_nodes"]
            },
            "evidence": ["Extracted vessel network graph", "Node connectivity"],
            "explanation": explanation,
            "model_status": self.model_status
        }
