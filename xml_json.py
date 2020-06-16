import xml.etree.ElementTree as ET

def jsonify(root):
    parsed = dict()
    for child in root:
        children = child.getchildren()
        if len(children) == 0:
            parsed[f"{child.tag}"] = [child.text]
        else:
            if f"{child.tag}" in parsed.keys():
                parsed[f"{child.tag}"].append(json(children))
            else:
                parsed[f"{child.tag}"] = [json(children)]
    for item in parsed.keys():
        if len(parsed[item]) == 1:
            parsed[item] = parsed[item][0]
    return parsed

def xml_json(filename):
    """
    Purpose:
        convert an xml file to a json file
    
    Inputs:
        filename: name of the file
    
    Return:
        dict: a dictionary in the json format, use json.dumps to save it properly
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    return {filename: jsonify(root)}

if __name__ == "__main__":
    print(xml_json("test1.xml"))