__kernel void computeDistanceMap(__global const int *grid, 
                                  __global float *distMap, 
                                  int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    if (grid[idx] == 1) {  // Obstacle
        distMap[idx] = 0.0f;
    } else {
        float minDist = FLT_MAX;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (grid[i * width + j] == 1) {
                    float dist = sqrt((float)((x-j)*(x-j) + (y-i)*(y-i)));
                    if (dist < minDist) minDist = dist;
                }
            }
        }
        distMap[idx] = minDist;
    }
}