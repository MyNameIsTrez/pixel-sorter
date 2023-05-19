void sort(int *a, int *b, int *c) {
   int tmp;

   if(*a > *b) {
      tmp = *a;
      *a = *b;
      *b = tmp;
   }
   if(*a > *c) {
      tmp = *a;
      *a = *c;
      *c = tmp;
   }
   if(*b > *c) {
      tmp = *b;
      *b = *c;
      *c = tmp;
   }
}
__kernel void medianFilter(
    __global float *img,
    __global float *result,
    __global int *width,
    __global int *height
) {
    int w = *width;
    int h = *height;
    int posx = get_global_id(1);
    int posy = get_global_id(0);
    int i = w * posy + posx;

    // Keeping the edge pixels the same
    if( posx == 0 || posy == 0 || posx == w - 1 || posy == h - 1 )
    {
        result[i] = img[i];
    }
    else
    {
        int pixel00 = img[i - 1 - w];
        int pixel01 = img[i - w];
        int pixel02 = img[i + 1 - w];
        int pixel10 = img[i - 1];
        int pixel11 = img[i];
        int pixel12 = img[i + 1];
        int pixel20 = img[i - 1 + w];
        int pixel21 = img[i + w];
        int pixel22 = img[i + 1 + w];

        //sort the rows
        sort( &(pixel00), &(pixel01), &(pixel02) );
        sort( &(pixel10), &(pixel11), &(pixel12) );
        sort( &(pixel20), &(pixel21), &(pixel22) );

        //sort the columns
        sort( &(pixel00), &(pixel10), &(pixel20) );
        sort( &(pixel01), &(pixel11), &(pixel21) );
        sort( &(pixel02), &(pixel12), &(pixel22) );

        //sort the diagonal
        sort( &(pixel00), &(pixel11), &(pixel22) );

        // median is the the middle value of the diagonal
        result[i] = pixel11;
    }
}
