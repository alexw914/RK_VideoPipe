#include "yolo_post.h"

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

inline static int32_t __clip(float val, float min, float max)
{
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
    float  dst_val = (f32 / scale) + zp;
    int8_t res     = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
{
    for (int i = 0; i < validCount; ++i) {
        if (order[i] == -1 || classIds[i] != filterId) continue;
        int n = order[i];
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[i] != filterId) continue;
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

            if (iou > threshold) order[j] = -1;
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices)
{
    float key;
    int key_index;
    int low = left;
    int high = right;
    if (left < right)
    {
        key_index = indices[left];
        key = input[left];
        while (low < high)
        {
            while (low < high && input[high] <= key)
            {
                high--;
            }
            input[low] = input[high];
            indices[low] = indices[high];
            while (low < high && input[low] >= key)
            {
                low++;
            }
            input[high] = input[low];
            indices[high] = indices[low];
        }
        input[low] = key;
        indices[low] = key_index;
        quick_sort_indice_inverse(input, left, low - 1, indices);
        quick_sort_indice_inverse(input, low + 1, right, indices);
    }
    return low;
}

void compute_dfl(float* tensor, int dfl_len, float* box){
    for (int b = 0; b < 4; b++){
        float exp_t[dfl_len];
        float exp_sum=0;
        float acc_sum=0;
        for (int i=0; i< dfl_len; i++){
            exp_t[i] = exp(tensor[i+b*dfl_len]);
            exp_sum += exp_t[i];
        }
        
        for (int i=0; i< dfl_len; i++){
            acc_sum += exp_t[i]/exp_sum *i;
        }
        box[b] = acc_sum;
    }
}

int PostProcessorBase::run(rknn_tensor_attr *output_attrs, rknn_output * outputs, 
                std::vector<DetectionResult> &results, LETTER_BOX& lb) {

    std::vector<float> filterBoxes, objProbs;
    std::vector<int> classId;

    int model_in_w = lb.target_width;
    int model_in_h = lb.target_height;

    int validCount = process_func(output_attrs, outputs, model_in_w, model_in_h, filterBoxes, objProbs, classId);

    // no object detect
    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; ++i) indexArray.push_back(i);

    // sort;
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    //nms
    std::set<int> class_set(std::begin(classId), std::end(classId));
    for (auto c : class_set) nms(validCount, filterBoxes, classId, indexArray, c, config.nms_threshold);

    int last_count = 0;
    /* box valid detect target */
    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1 || last_count >= 8) continue;
        int n = indexArray[i];
        struct DetectionResult obj;

        obj.id    = classId[n];
        obj.score = objProbs[i];
        obj.label = config.labels.at(classId[n]);

        obj.box.top    = w_reverse(filterBoxes[n * 4 + 0], lb);
        obj.box.left   = h_reverse(filterBoxes[n * 4 + 1], lb);
        obj.box.bottom = w_reverse(filterBoxes[n * 4 + 0] + filterBoxes[n * 4 + 2], lb);
        obj.box.right  = h_reverse(filterBoxes[n * 4 + 1] + filterBoxes[n * 4 + 3], lb);

        results.push_back(obj);
        last_count++;
    }
    return last_count;
}

int YOLOv5v7_Post::process_i8(int8_t *input, int *anchor, int grid_h, int grid_w, int stride,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId,
                      int32_t zp, float scale) {

    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = qnt_f32_to_affine(config.conf_threshold, zp, scale);
    
    // box_size = label_size + 5 (4point + 1obj_conf)
    for (int a = 0; a < 3; a++)
    {
        for (int i = 0; i < grid_h; i++)
        {
            for (int j = 0; j < grid_w; j++)
            {
                int8_t box_confidence = input[((config.labels.size() + 5) * a + 4) * grid_len + i * grid_w + j];
                if (box_confidence >= thres_i8)
                {
                    int offset = ((config.labels.size() + 5) * a) * grid_len + i * grid_w + j;
                    int8_t *in_ptr = input + offset;
                    float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                    float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                    float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                    float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                    box_x = (box_x + j) * (float)stride;
                    box_y = (box_y + i) * (float)stride;
                    box_w = box_w * box_w * (float)anchor[a * 2];
                    box_h = box_h * box_h * (float)anchor[a * 2 + 1];
                    box_x -= (box_w / 2.0);
                    box_y -= (box_h / 2.0);

                    int8_t maxClassProbs = in_ptr[5 * grid_len];
                    int maxClassId = 0;
                    for (int k = 1; k < config.labels.size(); ++k)
                    {
                        int8_t prob = in_ptr[(5 + k) * grid_len];
                        if (prob > maxClassProbs)
                        {
                            maxClassId = k;
                            maxClassProbs = prob;
                        }
                    }
                    if (maxClassProbs > thres_i8)
                    {
                        objProbs.push_back((deqnt_affine_to_f32(maxClassProbs, zp, scale)) * (deqnt_affine_to_f32(box_confidence, zp, scale)));
                        classId.push_back(maxClassId);
                        validCount++;
                        boxes.push_back(box_x);
                        boxes.push_back(box_y);
                        boxes.push_back(box_w);
                        boxes.push_back(box_h);
                    }
                }
            }
        }
    }
    return validCount;
}

int YOLOv6v8_Post::process_i8(int8_t *box_tensor, int32_t box_zp, float box_scale,
                      int8_t *score_tensor, int32_t score_zp, float score_scale,
                      int8_t *score_sum_tensor, int32_t score_sum_zp, float score_sum_scale,
                      int grid_h, int grid_w, int stride, int dfl_len,
                      std::vector<float> &boxes, std::vector<float> &objProbs, std::vector<int> &classId)
{
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t score_thres_i8 = qnt_f32_to_affine(config.conf_threshold, score_zp, score_scale);
    int8_t score_sum_thres_i8 = qnt_f32_to_affine(config.conf_threshold, score_sum_zp, score_sum_scale);

    for (int i = 0; i < grid_h; i++){
        for (int j = 0; j < grid_w; j++){
            int offset = i* grid_w + j;
            int max_class_id = -1;

            // 通过 score sum 起到快速过滤的作用
            if (score_sum_tensor != nullptr){
                if (score_sum_tensor[offset] < score_sum_thres_i8){
                    continue;
                }
            }

            int8_t max_score = -score_zp;
            for (int c = 0; c < config.labels.size(); c++){
                if ((score_tensor[offset] > score_thres_i8) && (score_tensor[offset] > max_score))
                {
                    max_score = score_tensor[offset];
                    max_class_id = c;
                }
                offset += grid_len;
            }

            // compute box
            if (max_score> score_thres_i8){
                offset = i * grid_w + j;
                float box[4];
                if (dfl_len>1){
                    /// dfl
                    float before_dfl[dfl_len*4];
                    for (int k = 0; k < dfl_len*4; k++){
                        before_dfl[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                    compute_dfl(before_dfl, dfl_len, box);
                }
                else {
                    for (int k = 0; k < 4; k++){
                        box[k] = deqnt_affine_to_f32(box_tensor[offset], box_zp, box_scale);
                        offset += grid_len;
                    }
                }

                float x1,y1,x2,y2,w,h;
                x1 = (-box[0] + j + 0.5) * stride;
                y1 = (-box[1] + i + 0.5) * stride;
                x2 = (box[2] + j + 0.5) * stride;
                y2 = (box[3] + i + 0.5) * stride;
                boxes.push_back(x1);
                boxes.push_back(y1);
                boxes.push_back(x2 - x1);
                boxes.push_back(y2 - y1);

                objProbs.push_back(deqnt_affine_to_f32(max_score, score_zp, score_scale));
                classId.push_back(max_class_id);
                validCount++;
            }
        }
    }
    return validCount;
}