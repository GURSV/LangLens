from langchain.tools import BaseTool
from transformers import BlipProcessor, BlipForConditionalGeneration, DetrImageProcessor, DetrForObjectDetection
from PIL import Image 
import torch

class ImageCaptionTool(BaseTool):
    name: str = "Image Captioner" 
    description: str = "Use this tool when given the path to an image that you would like to be described. " \
                       "It will return a simple caption describing the image."

    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        model_name = "Salesforce/blip-image-captioning-large"
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, return_tensors='pt').to(device)
        output = model.generate(**inputs, max_new_tokens=20)

        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async.")

class ObjectDetectionTool(BaseTool):
    name: str = "Object Detector" 
    description: str = "Use this tool when given the path to an image that you would like to detect objects. " \
                       "It will return a list of all detected objects. Each element in the list in the format: " \
                       "[x1, y1, x2, y2] class_name confidence_score."

    def _run(self, img_path: str) -> str:
        image = Image.open(img_path).convert('RGB')
        
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = ""
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[{}, {}, {}, {}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            detections += ' {}'.format(model.config.id2label[int(label)])
            detections += ' {}\n'.format(float(score))
        
        return detections

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async.")
    
class VisualQuestionAnsweringTool(BaseTool):
    name: str = "Visual Question Answering"  
    description: str = "Use this tool when you have a question about the content of an image. Pass the question and the image path in a single string, separated by a special delimiter."

    def _run(self, input_text: str) -> str:
        delimiter = "###"  # Defined a unique delimiter...
        try:
            question, img_path = input_text.split(delimiter)
        except ValueError:
            return "Error: Please format input as 'question ### image_path'."

        image = Image.open(img_path.strip()).convert('RGB')
        model_name = "Salesforce/blip-vqa-base"  
        device = "cpu"

        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

        inputs = processor(image, question.strip(), return_tensors='pt').to(device)
        output = model.generate(**inputs)

        answer = processor.decode(output[0], skip_special_tokens=True)
        return answer

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async.")
