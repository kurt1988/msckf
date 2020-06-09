#include "featureTracking.h"
#include "commonConfig.h"
#include "dataLoader.h"
#include "visualization.h"
#include "commonFuncs.h"
#include "msckf.h"
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

int main()
{
    DataLoader data;
    data.loadAndSynch();
    int init_frmID = 1000;
    int end_frmID = min(4000, data.getIMGInfo()->frameIDs.back());

    State state;
    state.initializeWithGT(data, init_frmID);

    int curr_frmID = init_frmID;

    string im_path = data.getIMGInfo()->getPath(curr_frmID);
    ImageArray im_array(im_path, curr_frmID);

    auto gt = data.getGT()->getData(data.getIMGInfo()->getTimestamp(curr_frmID));
    Vector3f gt_p = gt.p.front();
    Vector3f st_p = state.getFirstBlock()->p;
    float position_error = (gt_p - st_p).norm();
    cout << "slide_number: " << state.getSlides()->size() << "  error: "<< position_error <<endl;

    string outfilename = "/home/kurt/Documents/c_projects/vio/msckf/src/output.csv";
    ofstream outfile;
    outfile.open(outfilename, ios::out);
    outfile<<"frameID"<<','<<"p_error"<<endl;
    outfile<<curr_frmID<<','<<position_error<<endl;
    
    while (true)
    {
        curr_frmID++;
        cout<<"------------frmID: "<<curr_frmID<<endl;
        
        im_path = data.getIMGInfo()->getPath(curr_frmID);
        im_array.reset(im_path, curr_frmID);

        //state.getTracker()->trackPointByOpticalFlow(im_array);
        state.getTracker()->trackPointByORB(im_array);

        //cout<<"track_number: "<< state.getTracker()->getPointTrackList()->size()<<endl;
        
        if (state.decideKeyFrame())
        {
            cout<< "add Key Frame..."<<endl;
            double ts0 = data.getIMGInfo()->getTimestamp(state.getFirstBlock()->frameID);
            double ts1 = data.getIMGInfo()->getTimestamp(curr_frmID);
            auto imu_data = data.getIMU()->getData(ts0, ts1);

            state.predict(imu_data, curr_frmID);

            state.triangulation();

            state.visualCorrection();

            state.deleteLostPointTracks();
            state.deleteInversePointTracks();

            gt = data.getGT()->getData(ts1);

            gt_p = gt.p.front();
            st_p = state.getFirstBlock()->p;

            position_error = (gt_p - st_p).norm();
            cout << ".......................error: "<< position_error <<endl;
            outfile<<curr_frmID<<','<<position_error<<endl;

            //showTrack(state.getTracker(), 5, false, 1000);

        
        }
        

        if (curr_frmID > end_frmID - 10)
        {
            break;
        }
    }

    outfile.close();
}



