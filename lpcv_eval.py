# import os
import time
import xml.dom.minidom
import pathlib
import json

LPCV_CONTEST = pathlib.Path('/home/xilinx/jupyter_notebooks/lpcv_2021/')
IMG_DIR = LPCV_CONTEST / 'images'
RESULT_DIR = LPCV_CONTEST

# Return a batch of image dir  when `send` is called
class Team:
    def __init__(self, teamname, batch_size):
        # self.lpcv_contest = LPCV_CONTEST
        # self.img_dir = IMG_DIR                
        # self.result_path = RESULT_DIR        
        self._result_path = RESULT_DIR
        self.team_dir = LPCV_CONTEST  
        self.team_name = teamname 
        self.batch_size = batch_size  

        folder_list = [self.team_dir, self._result_path]
        for folder in folder_list:
            if not folder.is_dir():
                folder.mkdir()
        
        self.img_list = self.get_image_paths()
        self.batch_count = 0

    def get_image_paths(self):
        names_temp = [f for f in IMG_DIR.iterdir() if f.suffix == '.jpg']
        names_temp.sort(key= lambda x:int(x.stem))
        return names_temp

    # Returns list of images paths for next batch of images
    def get_next_batch(self):
        start_idx = self.batch_count * self.batch_size
        self.batch_count += 1
        end_idx = self.batch_count * self.batch_size

        if start_idx >= len(self.img_list):
            return None
        elif end_idx > len(self.img_list):
            return self.img_list[start_idx:]
        else:
            return self.img_list[start_idx:end_idx]
    
    def reset_batch_count(self):
        self.batch_count = 0

    def save_results_xml(self, result_rectangle, runtime, energy):
        doc = xml.dom.minidom.Document()
        root = doc.createElement('results')

        perf_e = doc.createElement('performance')
        
        # Runtime
        runtime_e = doc.createElement('runtime')
        runtime_e.appendChild(doc.createTextNode(str(runtime)))
        perf_e.appendChild(runtime_e)
        root.appendChild(runtime_e)

        # Energy
        energy_e = doc.createElement('energy')
        energy_e.appendChild(doc.createTextNode(str(energy)))
        perf_e.appendChild(energy_e)
        root.appendChild(energy_e)

        for i in range(len(result_rectangle)):
            image_e = root.appendChild(doc.createElement("image"))

            doc.appendChild(root)
            node_image_id = doc.createElement('image_id')
            node_image_id.appendChild(
                doc.createTextNode(str(result_rectangle[i][0])))
            node_label= doc.createElement('category_id')
            node_label.appendChild(
                doc.createTextNode(str(result_rectangle[i][5])))
            node_score = doc.createElement('score')
            node_score.appendChild(
                doc.createTextNode(str(result_rectangle[i][6])))
            node_bnd_box = doc.createElement('bbox')
            node_bnd_box_x = doc.createElement('x')
            node_bnd_box_x.appendChild(
                doc.createTextNode(str(result_rectangle[i][1])))
            node_bnd_box_y = doc.createElement('y')
            node_bnd_box_y.appendChild(
                doc.createTextNode(str(result_rectangle[i][2])))
            node_bnd_box_width = doc.createElement('width')
            node_bnd_box_width.appendChild(
                doc.createTextNode(str(result_rectangle[i][3])))
            node_bnd_box_height = doc.createElement('height')
            node_bnd_box_height.appendChild(
                doc.createTextNode(str(result_rectangle[i][4])))
            node_bnd_box.appendChild(node_bnd_box_x)
            node_bnd_box.appendChild(node_bnd_box_y)
            node_bnd_box.appendChild(node_bnd_box_width)
            node_bnd_box.appendChild(node_bnd_box_height)
            image_e.appendChild(node_image_id)
            image_e.appendChild(node_label)
            image_e.appendChild(node_bnd_box)
            image_e.appendChild(node_score)

        file_name = self._result_path / "results.xml"
        with open(file_name, 'w') as fp:
            doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

    def save_results_json(self, result_rectangle):
        json_file = open(self._result_path / str(self.team_name + ".json"), "w") 
        datas = []
        for i in range(len(result_rectangle)):  
            image_id = int(result_rectangle[i][0])
            label = int(result_rectangle[i][5])
            x = result_rectangle[i][1]
            y = result_rectangle[i][2]
            width = result_rectangle[i][3]
            height = result_rectangle[i][4]
            score = result_rectangle[i][6]
            data = {"image_id" : image_id, 
                    "category_id" : label, 
                    "bbox": [x, y, width, height],
                    "score": score,
                    }

            datas.append(data)

        json.dump(datas, json_file)
        json_file.close()
