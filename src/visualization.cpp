#include<iostream>
#include<string>
#include"featureTracking.h"
#include"visualization.h"

using namespace std;
using namespace cv;


void showTrack(Tracker * tracker, int ls_len, bool wait_char, int show_num)
{
    Mat im_mat;
    tracker->getImgTrackInfo()->gray_image.copyTo(im_mat);
    auto trackIDs = tracker->getImgTrackInfo()->pointIDs;
    auto frmID = tracker->getImgTrackInfo()->frameID;
    //string name = to_string(frmID);
    string name = "curr_img_track";
    RNG rng;
    cout<<"---1"<<endl;
    int cnt = 0;
    for (auto ptk : *tracker->getPointTrackList())
    {
        if (ptk.points.size() < ls_len)
            continue;
        auto color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        for (auto pt : ptk.points)
        {
            circle(im_mat, pt, 6, color);
        }
        cnt++;
        if (cnt > show_num)
            break;
    }

    imshow(name, im_mat);
    waitKey(0);
    
    if (wait_char)
    {
        cin.get();
    } 
}















