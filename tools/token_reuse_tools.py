import torch,json
import torchvision.transforms as transforms

MOT_CLASSES = ['person']

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

TRANSFORM_tiny = transforms.Compose([
            transforms.Resize((512,688)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

TRANSFORM_base = transforms.Compose([
            transforms.Resize((608,896)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def extract_bboxes_from_coco(json_path, image_name):
    """
    从COCO格式的json文件中提取指定图片的bbox信息。

    参数:
    - json_path: JSON文件的路径。
    - image_name: 要提取bbox信息的图片的名称。

    返回:
    - bboxes: 指定图片的所有bbox信息。
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 查找指定图片的ID
    image_id = None
    for image in data['images']:
        if image['file_name'] == image_name:
            image_id = image['id']
            break
    
    if image_id is None:
        raise (f"No image with the name {image_name} found.")


    # 提取与该图片ID关联的所有bbox信息
    bboxes = []
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']
            # 转换COCO的bbox格式 [x_top_left, y_top_left, width, height] 到 [x1, y1, x2, y2]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bboxes.append(bbox)

    return bboxes,image_id

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox).to('cpu')
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detections_to_coco_format(all_detections, all_image_ids):
    """
    Convert detections for multiple images to COCO-style format.
    
    Args:
    - all_detections (list of lists): A list where each entry is a list of detections for an image, 
                                      each detection in the format [x1, y1, x2, y2, confidence].
    - all_image_ids (list of ints): The IDs of the images.
    
    Returns:
    - coco_output (dict): The COCO-style formatted detections.
    """
    
    # Initialize data structure
    coco_output = {
        "images": [],
        "categories": [{"id": 1, "name": "object"}],  # Assuming a single category named 'object'
        "annotations": []
    }
    
    # Add image info to images field
    for image_id in all_image_ids:
        image_info = {"file_name": f"image_{image_id}.jpg", "id": image_id}
        coco_output["images"].append(image_info)

    # Add detections to annotations
    ann_id = 1  # start annotation ID from 1
    for detections, image_id in zip(all_detections, all_image_ids):
        for det in detections:
            x1, y1, x2, y2, confidence = det
            
            # Convert [x1, y1, x2, y2] format to [x, y, width, height]
            width = x2 - x1
            height = y2 - y1

            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x1, y1, width, height],
                "score": confidence
            }
            
            coco_output["annotations"].append(annotation)
            ann_id += 1

    return coco_output

def detections_to_cocojson(all_detections, all_image_ids, output_path):
    """
    Convert detections for multiple images to COCO-style JSON.
    
    Args:
    - all_detections (list of lists): A list where each entry is a list of detections for an image, 
                                      each detection in the format [x1, y1, x2, y2, confidence].
    - all_image_ids (list of ints): The IDs of the images.
    - output_path (str): Path to save the COCO-style JSON.
    """
    
    # Initialize data structure
    coco_output = {
        "images": [],
        "categories": [{"id": 1, "name": "object"}],  # Assuming a single category named 'object'
        "annotations": []
    }
    
    # Add image info to images field
    for image_id in all_image_ids:
        image_info = {"file_name": f"image_{image_id}.jpg", "id": image_id}
        coco_output["images"].append(image_info)

    # Add detections to annotations
    ann_id = 1  # start annotation ID from 1
    for detections, image_id in zip(all_detections, all_image_ids):
        for det in detections:
            x1, y1, x2, y2, confidence = det
            
            # Convert [x1, y1, x2, y2] format to [x, y, width, height]
            width = x2 - x1
            height = y2 - y1

            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x1, y1, width, height],
                "score": confidence
            }
            
            coco_output["annotations"].append(annotation)
            ann_id += 1
        
    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(coco_output, f)