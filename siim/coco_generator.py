import json

def make_json(df, image_set='train'):
    annotations = []
    images = []
    licenses = []
    categories = [{"categories":
                          [
                              { "id": 1,
                                "name": "opacity"}
                          ]
                      }]
    info = {
        "description": "SIIM Covid Dataset",
        "url": "http://kaggle.com",
        "version": "1.0",
        "year": 2021,
        "contributor": "SIIM",
        "date_created": "2021/07/01"
    }

    for idx, row in df.iterrows():
        image_id = row['id']
        file_name = row['id'][:12] + '.jpg'
        ann_str = row['label']
        ann_list_raw = ann_str.split(' ')
        no_ann = len(ann_list_raw) // 6

        # Images
        images.append({
            "file_name": file_name,
            "id": image_id
        })


        for i in range(no_ann):
            _temp = ann_list_raw[(i * 6): ((i + 1) * 6)]
            cat_name , cat_id, x, y, w,h = str(_temp[0]), str(_temp[1]), float(_temp[2]), float(_temp[3]), float(_temp[4]), float(_temp[5])
            if cat_name == 'none':
                continue
            else:
                ann_id = image_id + '_' + str(i)
                annotations.append({
                    "bbox": [x, y, w, h],
                    "category_id": 1,
                    "id": ann_id,
                    "image_id": image_id
                })

    content = {
        'info': info,
        'licenses': licenses,
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open('./{}.json'.format(image_set), 'w') as f:
        json.dump(content, f, ensure_ascii=False, indent=4)