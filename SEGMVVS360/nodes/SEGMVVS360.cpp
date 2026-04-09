/*
 * SEGMVVS: Semantic Equirectangular Gaussian Mixture Virtual Visual Servoing
 * Author: Hussein LOUBANI (Hussein.loubani@utbm.fr)
 * Institution: CIAD-UTBM
 * Date: December 2025
 */

//// ROS
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/UInt32.h>

//// Boost
#include <boost/filesystem.hpp>

//// ViSP
#include <visp/vpDisplayX.h>
#include <visp/vpExponentialMap.h>
#include <visp/vpImage.h>
#include <visp/vpImageConvert.h>
#include <visp/vpImageIo.h>
#include <visp/vpImageTools.h>
#include <visp/vpPlot.h>
#include <visp_bridge/3dpose.h>
#include <visp_bridge/image.h>

//// Std
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

//// Project services
#include "SEGMVVS360/building_segmentation.h"
#include "gaussian_mixture/ComputeGaussianMixture.h"

//// Global state and parameters
ros::Publisher robotVelocityPub;
ros::Publisher nextFrameTriggerPub;
ros::ServiceClient PGMClient;
ros::ServiceClient segClient;
gaussian_mixture::ComputeGaussianMixture PGMmsg;

// Image sizes and buffers
int imWidth = 0, imHeight = 0;
double imSizeFactor = 1.0;
vpImage<unsigned char> equiI, equiId, equiIdiff;
vpImage<vpRGBa> equiDrgb, IG, IGd, equiR, equiS, equiColorI;
vpImage<float> equiD, G, Gd, dGdu, dGdv, dGdb;

// Displays and plots
vpDisplayX equiDispI, equiDispId, equiDispIdiff, equiDispD, IGDisp, IGdDisp, equiDispR, equiDispS;
vpPlot plot;

// VS variables
vpPoseVector desiredRobotPose, currentRobotPose, initialRobotPose;
vpMatrix L;
vpColVector e, v;
int iter = 0;
int plot_iter = 0;
double gain_step1 = 0.5, gain_step2 = 0.5, lambda_step1 = 10.0, lambda_step2 = 1.0;
double gain_cur = 0.5, lambda_cur = 0.0;
bool vsStarted = false;

// Multi-frame
int totalFrames = 0;
int currentFrameIdx = 0;
int iterations_step1 = 15, iterations_step2 = 15;

// Misc
ros::Time t;
bool verbose = false;
int qSize = 30;
std::ofstream poseLog;
std::ofstream velLog;
std::ofstream timeLog;
ros::WallTime frame_start;
vpImage<unsigned char> dynMask;
bool hasDynMask = false;

// Output control
static std::string results_root;
static bool save_results = false;
static bool save_all_iterations = false;

// Topics
static std::string real_image_topic;
static std::string mask_topic;
static std::string color_topic;
static std::string get_pose_topic;
static std::string set_pose_topic;
static std::string depth_topic;
static std::string seg_service_name;

// Build subpaths under results_root
static std::string Dir(const std::string &sub) {
    return results_root.empty() ? std::string() : (results_root + "/" + sub);
}

//// Declarations
void cameraPosesInitialization();
void cameraImageRobotPoseCallback(const sensor_msgs::Image::ConstPtr &Imsg,
                                  const sensor_msgs::Image::ConstPtr &colorImsg,
                                  const geometry_msgs::PoseStamped::ConstPtr &robotPoseMsg,
                                  const sensor_msgs::Image::ConstPtr &equiDmsg);
void toRGBImage(vpImage<float> in, vpImage<vpRGBa> &out, float min = 0, float max = 0);
void computeEquirectangularGaussianMixtureInteraction(vpImage<float> &dGdu, vpImage<float> &dGdv,
                                                      vpImage<float> &dGdb, vpMatrix &L,
                                                      vpImage<float> &D);
void computeGaussianMixtureErrorVector(vpImage<float> G, vpImage<float> Gd, vpColVector &e,
                                       vpImage<float> D);
void displayAndFlush(vpImage<unsigned char> &I);
void displayAndFlush(vpImage<vpRGBa> &I);
bool computeGMM(const vpImage<unsigned char> &I, double beta,
                vpImage<float> *G_out, vpImage<float> *dGdu_out, vpImage<float> *dGdv_out,
                vpImage<float> *dGdb_out);
void saveInitialImagesForFrame(int idx);
void saveFinalImagesForFrame(int idx);
void saveAllIterationImages(int iter);
void prepareOutputDirectories();
void preprocessMask(vpImage<unsigned char> &I, const vpImage<unsigned char> &dyn, bool hasDyn);
void updateDesiredGMM(double beta);
void displayPanels();
void allocateImageBuffers();
void initializeDisplayWindows();
void initializePlots();

vpImage<float> imageMsgToVpImageFloat(const sensor_msgs::Image &Imsg);

//// Main
int main(int argc, char **argv) {
    // Initialize ROS node
    ros::init(argc, argv, "segmvvs360_node", ros::init_options::NoSigintHandler);
    if (ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
        ros::console::notifyLoggerLevelsChanged();
    }
    ros::NodeHandle nh("~");

    nh.param<std::string>("real_image_topic", real_image_topic, "/real_image");
    nh.param<std::string>("mask_topic",       mask_topic,       "/skyline/mask");
    nh.param<std::string>("color_topic",      color_topic,      "/skyline/color");
    nh.param<std::string>("get_pose_topic",   get_pose_topic,   "/skyline_controller/pose");
    nh.param<std::string>("set_pose_topic",   set_pose_topic,   "/skyline_controller/set_pose");
    nh.param<std::string>("depth_topic",      depth_topic,      "/skyline/depth");
    nh.param<std::string>("seg_service_name", seg_service_name, "/building_segmentation");

    // Total frames set by publisher
    nextFrameTriggerPub = nh.advertise<std_msgs::UInt32>("/next_frame_trigger", 1, false);

    // wait for publisher to set /total_targets
    totalFrames = 0;
    while (ros::ok() && totalFrames <= 0) {
        ros::param::get("/total_targets", totalFrames);
        ros::Duration(0.1).sleep();
    }
    ROS_INFO_STREAM("[VS] totalFrames = " << totalFrames);

    // ROS Parameters
    verbose          = nh.param("verbose",           1);
    gain_step1       = nh.param("gain_step1",      0.5);
    gain_step2       = nh.param("gain_step2",      0.5);
    lambda_step1     = nh.param("lambda_step1",   10.0);
    lambda_step2     = nh.param("lambda_step2",    1.0);
    iterations_step1 = nh.param("iterations_step1", 15);
    iterations_step2 = nh.param("iterations_step2", 15);
    imSizeFactor     = nh.param("im_size_factor",  1.0);
    imWidth          = nh.param("im_width",         320) * imSizeFactor;
    imHeight         = nh.param("im_height",        160) * imSizeFactor;

    // Output control
    results_root = nh.param<std::string>("results_root", std::string());
    save_results = nh.param("save_results", false);
    save_all_iterations = nh.param("save_all_iterations", false);
    if (save_results && results_root.empty()) {
        ROS_WARN("save_results is true but results_root is empty, disabling save_results");
        save_results = false;
    }

    // Initialize Gaussian Mixture service client
    std::string pgm_service;
    nh.param<std::string>("pgm_service_name", pgm_service, "/compute_equirectangular_gaussian_mixture_lambda");
    ROS_INFO_STREAM("Waiting for PGM service: " << pgm_service);
    ros::service::waitForService(pgm_service);
    PGMClient = nh.serviceClient<gaussian_mixture::ComputeGaussianMixture>(pgm_service);

    // Initialize segmentation service client
    ROS_INFO_STREAM("Waiting for segmentation service: " << seg_service_name);
    ros::service::waitForService(seg_service_name);
    segClient = nh.serviceClient<SEGMVVS360::building_segmentation>(seg_service_name);

    // Initialize robot velocity publisher
    robotVelocityPub = nh.advertise<geometry_msgs::Pose>(set_pose_topic, 1);

    // Initialize image and pose subscribers
    message_filters::Subscriber<sensor_msgs::Image> maskSub;
    message_filters::Subscriber<sensor_msgs::Image> colorSub;
    message_filters::Subscriber<geometry_msgs::PoseStamped> robotPoseSub;
    message_filters::Subscriber<sensor_msgs::Image> depthSub;

    maskSub.subscribe(nh, mask_topic, qSize);
    colorSub.subscribe(nh, color_topic, qSize);
    robotPoseSub.subscribe(nh, get_pose_topic, qSize);
    depthSub.subscribe(nh, depth_topic, qSize);

    // Synchronize camera and pose topics
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped, sensor_msgs::Image> camerasSyncPolicy;
    message_filters::Synchronizer<camerasSyncPolicy> camerasSynchronizer(
        camerasSyncPolicy(qSize), maskSub, colorSub, robotPoseSub, depthSub);
    camerasSynchronizer.registerCallback(boost::bind(&cameraImageRobotPoseCallback, _1, _2, _3, _4));

    // Allocate image buffers
    allocateImageBuffers();

    // Initialize display windows
    initializeDisplayWindows();

    // Initialize diagnostic plots
    initializePlots();

    // Always save pose log
    poseLog.open(results_root.empty() ? "pose_log.txt" : Dir("pose_log.txt").c_str());

    // Prepare outputs if requested
    if (save_results) {
        prepareOutputDirectories();

        velLog.open(Dir("vel_log.txt").c_str());
        velLog << "vx vy vz wx wy wz lambda\n";
        timeLog.open(Dir("time_log.txt").c_str());
        timeLog << "stamp_sec\n";
    }

    // Start VS
    v.resize(6);
    cameraPosesInitialization();
    iter = 0;
    vsStarted = true;
    t = ros::Time::now();

    // Spin ROS node
    ros::spin();

    if (poseLog.is_open()) poseLog.close();
    if (velLog.is_open()) velLog.close();
    if (timeLog.is_open()) timeLog.close();
}

//// Initialization of desired & initial pose
void cameraPosesInitialization() {
    ros::NodeHandle nh("~");

    // Countdown before initialization
    ROS_INFO("Initializing camera pose setup...");
    for (int i = 3; i > 0; --i) {
        ROS_INFO_STREAM("Starting in " << i << " seconds...");
        ros::Duration(1.0).sleep();
    }

    // Fetch real image and run segmentation
    currentFrameIdx = 0;
    std_msgs::UInt32 req;
    req.data = currentFrameIdx;
    nextFrameTriggerPub.publish(req);

    ROS_INFO_STREAM("Waiting for real image from topic: " << real_image_topic << "...");
    sensor_msgs::Image::ConstPtr realImageMsg = ros::topic::waitForMessage<sensor_msgs::Image>(real_image_topic);

    vpImage<vpRGBa> Itmp(realImageMsg->height, realImageMsg->width);
    Itmp = visp_bridge::toVispImageRGBa(*realImageMsg);

    vpImageTools::resize(Itmp, equiR, imWidth, imHeight);
    displayAndFlush(equiR);

    SEGMVVS360::building_segmentation srv;

    srv.request.real_rgb_image = *realImageMsg;

    ROS_INFO("Calling segmentation service...");
    if (!segClient.call(srv)) {
        ROS_ERROR("Segmentation service call failed. Using raw image as fallback.");
        equiId = visp_bridge::toVispImage(*realImageMsg);
    } else if (!srv.response.success) {
        ROS_WARN_STREAM("Segmentation failed: " << srv.response.message);
        equiId = visp_bridge::toVispImage(*realImageMsg);
    } else {
        equiS = visp_bridge::toVispImageRGBa(srv.response.semantic_segmentation_image);
        equiId = visp_bridge::toVispImage(srv.response.real_binary_mask_image);

        // dynamic occlusion mask
        dynMask = visp_bridge::toVispImage(srv.response.dynamic_occlusion_image);
        hasDynMask = true;
    }

    // Display segmentation result
    displayAndFlush(equiId);
    displayAndFlush(equiS);
    ros::spinOnce();

    // Compute Desired Gaussian Mixture
    lambda_cur = lambda_step1;
    if (computeGMM(equiId, lambda_step1, &Gd, nullptr, nullptr, nullptr)) {
        if (Gd.getHeight() == 0 || Gd.getWidth() == 0) {
            ROS_ERROR("Gaussian Mixture response is empty.");
            return;
        }
        ROS_INFO("Gaussian Mixture computation succeeded.");
        toRGBImage(Gd, IGd);
    } else {
        ROS_ERROR("Gaussian Mixture service call failed.");
        return;
    }

    // Initial pose selection
    ROS_INFO("Please move the camera to the Initial pose and click on Real Image...");
    do {
        vpImageTools::imageDifference(equiI, equiId, equiIdiff);
        displayAndFlush(equiI);
        displayAndFlush(equiIdiff);
        displayAndFlush(equiId);
        displayAndFlush(IGd);
        ros::spinOnce();
    } while (!vpDisplay::getClick(equiR, false));

    initialRobotPose = currentRobotPose;

    ROS_INFO("Ready. Click on window Real Image to start Visual Servoing...");
    do {
        ros::spinOnce();
    } while (!vpDisplay::getClick(equiR, false));

    // Log desired pose
    desiredRobotPose = currentRobotPose;  // desired = pose at start
    
    if (poseLog.is_open()) {
        poseLog << desiredRobotPose[0] << " " << desiredRobotPose[1] << " " << desiredRobotPose[2] << " "
                << desiredRobotPose[3] << " " << desiredRobotPose[4] << " " << desiredRobotPose[5] << "\n";
        poseLog.flush();
    }
}

//// Main callback function
void cameraImageRobotPoseCallback(const sensor_msgs::Image::ConstPtr &equiImsg,
                                  const sensor_msgs::Image::ConstPtr &equiColorImsg,
                                  const geometry_msgs::PoseStamped::ConstPtr &robotPoseMsg,
                                  const sensor_msgs::Image::ConstPtr &equiDmsg) {
    // Convert ROS images to ViSP format and resize
    if (imSizeFactor != 1.0) {
        vpImageTools::resize(visp_bridge::toVispImage(*equiImsg), equiI, imWidth, imHeight);
        vpImageTools::resize(visp_bridge::toVispImageRGBa(*equiColorImsg), equiColorI, imWidth, imHeight);
        vpImageTools::resize(imageMsgToVpImageFloat(*equiDmsg), equiD, imWidth, imHeight);
    } else {
        equiI = visp_bridge::toVispImage(*equiImsg);
        equiColorI = visp_bridge::toVispImageRGBa(*equiColorImsg);
        equiD = imageMsgToVpImageFloat(*equiDmsg);
    }
    preprocessMask(equiI, dynMask, hasDynMask);

    // Extract the current robot pose
    currentRobotPose = vpPoseVector(visp_bridge::toVispHomogeneousMatrix(robotPoseMsg->pose));

    if (!vsStarted) return;
    if (equiImsg->header.stamp <= t) return;

    // Step 1
    if (iter == 0) {
        frame_start = ros::WallTime::now();
        gain_cur = gain_step1;
        lambda_cur = lambda_step1;
        // refresh desired with step-1 lambda
        updateDesiredGMM(lambda_step1);
    }

    // Step 2
    if (iter == iterations_step1) {
        gain_cur = gain_step2;
        lambda_cur = lambda_step2;
        updateDesiredGMM(lambda_step2);
    }

    // Compute current GMM and gradients
    if (!computeGMM(equiI, lambda_cur, &G, &dGdu, &dGdv, &dGdb)) {
        ROS_ERROR("GMM service call failed during VS loop, skipping iteration.");
        return;
    }

    // Displays
    displayPanels();

    // Save all iterations of the frame if enabled
    if (save_all_iterations && save_results) {
        saveAllIterationImages(iter);
    }

    // Save initial image of all frames
    if (iter == 0 && save_results) {
        const int idx = currentFrameIdx + 1;
        saveInitialImagesForFrame(idx);
    }

    // Compute interaction matrix from image gradients
    computeEquirectangularGaussianMixtureInteraction(dGdu, dGdv, dGdb, L, equiD);

    // Compute error vector between current and desired images
    computeGaussianMixtureErrorVector(G, Gd, e, equiD);

    // Gauss-Newton Control law
    vpColVector vl(7);
    vl = -gain_cur * L.pseudoInverseEigen3() * e;
    v[0] = vl[0];
    v[1] = vl[1];
    v[2] = vl[2];
    v[3] = vl[3];
    v[4] = vl[4];
    v[5] = vl[5];
    lambda_cur += vl[6];

    // Time and velocity logs
    if (velLog.is_open()) {
        velLog << v[0] << " " << v[1] << " " << v[2] << " " << v[3] << " " << v[4] << " " << v[5]
               << " " << lambda_cur << "\n";
        velLog.flush();
    }
    if (timeLog.is_open()) {
        timeLog << std::fixed << std::setprecision(9) << equiImsg->header.stamp.toSec() << "\n";
        timeLog.flush();
    }

    // Publish updated camera pose and time stamp
    robotVelocityPub.publish(visp_bridge::toGeometryMsgsPose(vpHomogeneousMatrix(currentRobotPose) *
                                                             vpExponentialMap::direct(v, 1.0)));
    t = ros::Time::now();

    // Plot error, pose, and velocities
    plot.plot(0, 0, plot_iter, e.sumSquare());
    plot.plot(1, 0, plot_iter, lambda_cur);

    // Display debug information if enabled (VERBOSE)
    if (verbose) {
        ROS_DEBUG("Iteration: %d", iter);
        ROS_DEBUG("Velocities: %f %f %f %f %f %f", v[0], v[1], v[2], v[3], v[4], v[5]);
        ROS_DEBUG("Current Pose: %f %f %f %f %f %f", currentRobotPose[0], currentRobotPose[1],
                  currentRobotPose[2], currentRobotPose[3], currentRobotPose[4], currentRobotPose[5]);
        ROS_DEBUG("Feature error: %f", e.sumSquare());
        ROS_DEBUG("Current lambda: %f", lambda_cur);
    }

    // Iteration incrementation
    ++iter;
    ++plot_iter;

    // Current frame completed
    if (iter >= (iterations_step1 + iterations_step2)) {
        const double sec = (ros::WallTime::now() - frame_start).toSec();
        ROS_INFO_STREAM("[VS] Frame " << (currentFrameIdx + 1) << " time = " << sec << " s");

        if (poseLog.is_open()) {
            poseLog << currentRobotPose[0] << " " << currentRobotPose[1] << " "
                    << currentRobotPose[2] << " " << currentRobotPose[3] << " "
                    << currentRobotPose[4] << " " << currentRobotPose[5] << "\n";
            poseLog.flush();
        }

        toRGBImage(equiD, equiDrgb, 0, 100);
        toRGBImage(G, IG);
        toRGBImage(Gd, IGd);

        if (save_results) {
            const int idx = currentFrameIdx + 1;
            saveFinalImagesForFrame(idx);
        }

        // Next frame
        currentFrameIdx++;
        if (currentFrameIdx >= totalFrames) {
            ROS_INFO("[VS] All frames processed. Stopping.");
            vsStarted = false;
            return;
        } else {
            ROS_INFO_STREAM("[VS] Requesting frame " << (currentFrameIdx + 1));
        }

        // Trigger next image by index
        std_msgs::UInt32 req;
        req.data = currentFrameIdx;
        nextFrameTriggerPub.publish(req);

        // ROS_INFO("Waiting for /real_image (frame %u)...", req.data + 1);
        sensor_msgs::Image::ConstPtr newImageMsg =
            ros::topic::waitForMessage<sensor_msgs::Image>(real_image_topic);

        // Save the RGB image into equiR
        vpImage<vpRGBa> tmpRGB = visp_bridge::toVispImageRGBa(*newImageMsg);
        vpImageTools::resize(tmpRGB, equiR, imWidth, imHeight);

        // ROS_INFO("Calling segmentation for new frame...");
        SEGMVVS360::building_segmentation srv;
        srv.request.real_rgb_image = *newImageMsg;

        if (!segClient.call(srv) || !srv.response.success) {
            ROS_WARN("Segmentation failed. Using raw image.");
            equiId = visp_bridge::toVispImage(*newImageMsg);
        } else {
            equiS = visp_bridge::toVispImageRGBa(srv.response.semantic_segmentation_image);
            equiId = visp_bridge::toVispImage(srv.response.real_binary_mask_image);
            dynMask = visp_bridge::toVispImage(srv.response.dynamic_occlusion_image);
            hasDynMask = true;
        }

        lambda_cur = lambda_step1;
        gain_cur   = gain_step1;

        displayAndFlush(equiS);
        displayAndFlush(equiId);
        toRGBImage(Gd, IGd);
        displayAndFlush(IGd);
        displayAndFlush(equiR);

        frame_start = ros::WallTime::now();

        // Reset iteration counter for new frame
        iter = 0;
    }
}

void computeEquirectangularGaussianMixtureInteraction(vpImage<float> &dGdu_, vpImage<float> &dGdv_,
                                                      vpImage<float> &dGdb_, vpMatrix &L,
                                                      vpImage<float> &D) {
    L.clear();
    L.resize(dGdu_.getWidth() * dGdu_.getHeight(), 7);

    float *ptrdGdu = dGdu_.bitmap;
    float *ptrdGdv = dGdv_.bitmap;
    float *ptrdGdb = dGdb_.bitmap;

    double sX, sY, sZ, X, Y, Z, x, y, depth;
    double au = dGdu_.getCols() / (2.0 * M_PI);
    double av = dGdu_.getRows() / M_PI;
    double u0 = dGdu_.getCols() / 2.0;
    double v0 = dGdu_.getRows() / 2.0;

    vpRowVector dGdu__(2);
    vpMatrix dudx(2, 2, 0.0);
    vpMatrix dxdX(2, 3, 0.0);
    vpMatrix dXdr(3, 6, 0.0);
    vpRowVector dGdr(6);

    dudx[0][0] = au;
    dudx[0][1] = 0.0;
    dudx[1][0] = 0.0;
    dudx[1][1] = av;

    int i = 0;
    for (int v = 0; v < dGdu_.getHeight(); v++) {
        for (int u = 0; u < dGdu_.getWidth(); u++, i++) {
            x = ((double)(u)-u0) / au;
            y = ((double)(v)-v0) / av;

            sX = cos(y) * sin(x);
            sY = sin(y);
            sZ = cos(y) * cos(x);

            depth = D[v][u];

            if (depth > 100) depth = 10000;

            X = sX * depth;
            Y = sY * depth;
            Z = sZ * depth;

            dGdu__[0] = ptrdGdu[u + v * dGdu_.getCols()];
            dGdu__[1] = ptrdGdv[u + v * dGdu_.getCols()];

            double X2 = X * X, Y2 = Y * Y, Z2 = Z * Z;
            double X2pZ2 = X2 + Z2;
            double srX2pZ2 = sqrt(X2pZ2);
            double X2pY2pZ2 = X2pZ2 + Y2;

            if (X2pZ2 < 1e-8) {
                dxdX.resize(2, 3, true);
            } else {
                dxdX[0][0] = Z / X2pZ2;
                dxdX[0][1] = 0.;
                dxdX[0][2] = -X / X2pZ2;
                dxdX[1][0] = -X * Y / (srX2pZ2 * X2pY2pZ2);
                dxdX[1][1] = srX2pZ2 / X2pY2pZ2;
                dxdX[1][2] = -Y * Z / (srX2pZ2 * X2pY2pZ2);
            }

            dXdr[0][0] = -1.0;
            dXdr[0][1] = 0.0;
            dXdr[0][2] = 0.0;
            dXdr[1][0] = 0.0;
            dXdr[1][1] = -1.0;
            dXdr[1][2] = 0.0;
            dXdr[2][0] = 0.0;
            dXdr[2][1] = 0.0;
            dXdr[2][2] = -1.0;

            dXdr[0][3] = 0.0;
            dXdr[0][4] = -Z;
            dXdr[0][5] = Y;
            dXdr[1][3] = Z;
            dXdr[1][4] = 0.0;
            dXdr[1][5] = -X;
            dXdr[2][3] = -Y;
            dXdr[2][4] = X;
            dXdr[2][5] = 0.0;

            dGdr = -dGdu__ * dudx * dxdX * dXdr;
            L.insert(dGdr, i, 0);
            L[i][6] = static_cast<double>(ptrdGdb[u + v * dGdu_.getCols()]);
        }
    }
}

void computeGaussianMixtureErrorVector(vpImage<float> G, vpImage<float> Gd, vpColVector &e,
                                       vpImage<float> D) {
    e.resize(G.getWidth() * G.getHeight());

    float *ptrG = G.bitmap;
    float *ptrGd = Gd.bitmap;

    int i = 0;
    for (int v = 0; v < G.getHeight(); v++) {
        for (int u = 0; u < G.getWidth(); u++, i++) {
            float depth = D[v][u];
            if (depth <= 0.0f || std::isnan(depth)) {
                e[i] = 0.0;
            } else {
                e[i] = ptrG[u + v * G.getCols()] - ptrGd[u + v * G.getCols()];
            }
        }
    }
}

void displayAndFlush(vpImage<unsigned char> &I) {
    vpDisplay::display(I);
    vpDisplay::flush(I);
}

void displayAndFlush(vpImage<vpRGBa> &I) {
    vpDisplay::display(I);
    vpDisplay::flush(I);
}

bool computeGMM(const vpImage<unsigned char> &I, double beta,
                vpImage<float> *G_out, vpImage<float> *dGdu_out, vpImage<float> *dGdv_out,
                vpImage<float> *dGdb_out) {
    PGMmsg.request.I = visp_bridge::toSensorMsgsImage(I);
    PGMmsg.request.beta = beta;

    if (!PGMClient.call(PGMmsg)) return false;

    if (G_out) *G_out = imageMsgToVpImageFloat(PGMmsg.response.G);
    if (dGdu_out) *dGdu_out = imageMsgToVpImageFloat(PGMmsg.response.dGdu);
    if (dGdv_out) *dGdv_out = imageMsgToVpImageFloat(PGMmsg.response.dGdv);
    if (dGdb_out) *dGdb_out = imageMsgToVpImageFloat(PGMmsg.response.dGdb);
    return true;
}

void preprocessMask(vpImage<unsigned char> &I, const vpImage<unsigned char> &dyn, bool hasDyn) {
    // invert
    for (unsigned int v = 0; v < I.getHeight(); ++v)
        for (unsigned int u = 0; u < I.getWidth(); ++u) I[v][u] = 255 - I[v][u];

    // dynamic occluders
    if (hasDyn) {
        if (dyn.getWidth() != I.getWidth() || dyn.getHeight() != I.getHeight()) {
            vpImage<unsigned char> dynResized;
            vpImageTools::resize(dyn, dynResized, I.getWidth(), I.getHeight());
            for (unsigned int v = 0; v < I.getHeight(); ++v)
                for (unsigned int u = 0; u < I.getWidth(); ++u)
                    if (dynResized[v][u]) I[v][u] = 0;
        } else {
            for (unsigned int v = 0; v < I.getHeight(); ++v)
                for (unsigned int u = 0; u < I.getWidth(); ++u)
                    if (dyn[v][u]) I[v][u] = 0;
        }
    }
}

void updateDesiredGMM(double beta) {
    computeGMM(equiId, beta, &Gd, nullptr, nullptr, nullptr);
    toRGBImage(Gd, IGd);
}

void saveInitialImagesForFrame(int idx) {
    vpImageIo::write(equiR, Dir("real_rgb_images/real_rgb_") + std::to_string(idx) + ".jpg");
    vpImageIo::write(equiS, Dir("real_semantic_segmentation_images/real_semantic_segmentation_") +
                                std::to_string(idx) + ".jpg");
    vpImageIo::write(equiId,
                     Dir("real_binary_masks/real_binary_mask_") + std::to_string(idx) + ".jpg");
    vpImageIo::write(IGd, Dir("initial_real_gaussian_mixtures/initial_real_gaussian_mixture_") +
                              std::to_string(idx) + ".jpg");
    vpImageIo::write(equiColorI,
                     Dir("initial_synth_albedo/initial_albedo_") + std::to_string(idx) + ".jpg");
    vpImageIo::write(equiI, Dir("initial_synth_binary_masks/initial_synth_binary_mask_") +
                                std::to_string(idx) + ".jpg");
    vpImageIo::write(IG, Dir("initial_synth_gaussian_mixtures/initial_synth_gaussian_mixture_") +
                             std::to_string(idx) + ".jpg");
    vpImageIo::write(equiDrgb, Dir("initial_synth_depth_images/initial_depth_image_") +
                                   std::to_string(idx) + ".jpg");
    vpImageIo::write(
        equiIdiff, Dir("initial_ediff_images/initial_ediff_image_") + std::to_string(idx) + ".jpg");
}

void saveFinalImagesForFrame(int idx) {
    vpImageIo::write(IGd, Dir("final_real_gaussian_mixtures/final_real_gaussian_mixture_") +
                              std::to_string(idx) + ".jpg");
    vpImageIo::write(equiColorI,
                     Dir("final_synth_albedo/final_albedo_") + std::to_string(idx) + ".jpg");
    vpImageIo::write(equiI, Dir("final_synth_binary_masks/final_synth_binary_mask_") +
                                std::to_string(idx) + ".jpg");
    vpImageIo::write(IG, Dir("final_synth_gaussian_mixtures/final_gaussian_mixture_") +
                             std::to_string(idx) + ".jpg");
    vpImageIo::write(equiDrgb, Dir("final_synth_depth_images/final_depth_image_") +
                                   std::to_string(idx) + ".jpg");
    vpImageIo::write(equiIdiff,
                     Dir("final_ediff_images/final_ediff_image_") + std::to_string(idx) + ".jpg");
}

void saveAllIterationImages(int iter) {
    std::ostringstream s;
    s << "_iter_" << std::setw(4) << std::setfill('0') << iter;
    vpImage<vpRGBa> Drgb;
    toRGBImage(equiD, Drgb, 0, 100);
    vpImageIo::write(equiI,
                     Dir("all_iterations/synth_binary_masks/synth_binary_mask") + s.str() + ".jpg");
    vpImageIo::write(IGd, Dir("all_iterations/real_gaussian_mixtures/real_gaussian_mixture_") +
                              s.str() + ".jpg");
    vpImageIo::write(IG, Dir("all_iterations/synth_gaussian_mixtures/synth_gaussian_mixture_") +
                             s.str() + ".jpg");
    vpImageIo::write(Drgb,
                     Dir("all_iterations/synth_depth_images/depth_image_") + s.str() + ".jpg");
    vpImageIo::write(equiIdiff, Dir("all_iterations/ediff_images/ediff_image") + s.str() + ".jpg");
    vpImageIo::write(equiColorI,
                     Dir("all_iterations/albedo_images/albedo_image") + s.str() + ".jpg");
}

void prepareOutputDirectories() {
    boost::filesystem::create_directories(results_root);
    boost::filesystem::create_directories(Dir("real_rgb_images"));
    boost::filesystem::create_directories(Dir("real_binary_masks"));
    boost::filesystem::create_directories(Dir("initial_real_gaussian_mixtures"));
    boost::filesystem::create_directories(Dir("final_real_gaussian_mixtures"));
    boost::filesystem::create_directories(Dir("initial_synth_albedo"));
    boost::filesystem::create_directories(Dir("final_synth_albedo"));
    boost::filesystem::create_directories(Dir("initial_synth_binary_masks"));
    boost::filesystem::create_directories(Dir("final_synth_binary_masks"));
    boost::filesystem::create_directories(Dir("initial_synth_gaussian_mixtures"));
    boost::filesystem::create_directories(Dir("final_synth_gaussian_mixtures"));
    boost::filesystem::create_directories(Dir("initial_synth_depth_images"));
    boost::filesystem::create_directories(Dir("final_synth_depth_images"));
    boost::filesystem::create_directories(Dir("initial_ediff_images"));
    boost::filesystem::create_directories(Dir("final_ediff_images"));
    boost::filesystem::create_directories(Dir("real_semantic_segmentation_images"));

    if (save_all_iterations) {
        boost::filesystem::create_directories(Dir("all_iterations/synth_binary_masks"));
        boost::filesystem::create_directories(Dir("all_iterations/real_gaussian_mixtures"));
        boost::filesystem::create_directories(Dir("all_iterations/synth_gaussian_mixtures"));
        boost::filesystem::create_directories(Dir("all_iterations/synth_depth_images"));
        boost::filesystem::create_directories(Dir("all_iterations/ediff_images"));
        boost::filesystem::create_directories(Dir("all_iterations/albedo_images"));
    }
}

void displayPanels() {
    vpImageTools::imageDifference(equiI, equiId, equiIdiff);
    displayAndFlush(equiI);
    displayAndFlush(equiId);
    displayAndFlush(equiIdiff);
    toRGBImage(equiD, equiDrgb, 0, 100);
    displayAndFlush(equiDrgb);
    toRGBImage(G, IG);
    displayAndFlush(IG);
    toRGBImage(Gd, IGd);
    displayAndFlush(IGd);
}

void allocateImageBuffers() {
    equiI.resize(imHeight, imWidth);
    equiId.resize(imHeight, imWidth);
    equiIdiff.resize(imHeight, imWidth);
    equiD.resize(imHeight, imWidth);
    equiDrgb.resize(imHeight, imWidth);
    IG.resize(imHeight, imWidth);
    IGd.resize(imHeight, imWidth);
    G.resize(imHeight, imWidth);
    Gd.resize(imHeight, imWidth);
    dGdu.resize(imHeight, imWidth);
    dGdv.resize(imHeight, imWidth);
    dGdb.resize(imHeight, imWidth);
    equiR.resize(imHeight, imWidth);
    equiS.resize(imHeight, imWidth);
}

void initializeDisplayWindows() {
    equiDispR.init(equiR, 0, 27, "Real Image");
    equiDispS.init(equiS, imWidth, 27, "Semantic Segmentation");
    equiDispId.init(equiId, imWidth * 2, 27, "Desired Binary Mask");
    equiDispI.init(equiI, imWidth * 2, 27 + 37 + imHeight, "Binary Mask");
    equiDispIdiff.init(equiIdiff, imWidth * 2, 27 + (37 + imHeight) * 2, "Image Difference");
    IGdDisp.init(IGd, imWidth * 3, 27, "Desired Gaussian Mixture");
    IGDisp.init(IG, imWidth * 3, 27 + 37 + imHeight, "Gaussian Mixture");
    equiDispD.init(equiDrgb, imWidth * 3, 27 + (37 + imHeight) * 2, "Depth Image");
}

void initializePlots() {
    // Base window
    plot.init(2, imHeight * 2 + 37, imWidth * 2, 0, 37 + 27 + imHeight, "Error");

    // Graph 0: Error
    plot.initGraph(0, 1);
    plot.setTitle(0, "Feature error (||e||^2)");
    plot.setLegend(0, 0, "sumSquare(e)");

    // Graph 1: Lambda
    plot.initGraph(1, 1);
    plot.setTitle(1, "Gaussian extent λ");
    plot.setLegend(1, 0, "lambda");
}

float RGB[3][64] = {
    {0,        0,        0,        0,        0,        0,        0,        0,
     0,        0,        0,        0,        0,        0,        0,        0,
     0,        0,        0,        0,        0,        0,        0,        0,
     15.9375,  31.8750,  47.8125,  63.7500,  79.6875,  95.6250,  111.5625, 127.5000,
     143.4375, 159.3750, 175.3125, 191.2500, 207.1875, 223.1250, 239.0625, 255.0000,
     255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000,
     255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000,
     239.0625, 223.1250, 207.1875, 191.2500, 175.3125, 159.3750, 143.4375, 127.5000},
    {0,        0,        0,        0,        0,        0,        0,        0,
     15.9375,  31.8750,  47.8125,  63.7500,  79.6875,  95.6250,  111.5625, 127.5000,
     143.4375, 159.3750, 175.3125, 191.2500, 207.1875, 223.1250, 239.0625, 255.0000,
     255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000,
     255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000,
     239.0625, 223.1250, 207.1875, 191.2500, 175.3125, 159.3750, 143.4375, 127.5000,
     111.5625, 95.6250,  79.6875,  63.7500,  47.8125,  31.8750,  15.9375,  0,
     0,        0,        0,        0,        0,        0,        0,        0},
    {143.4375, 159.3750, 175.3125, 191.2500, 207.1875, 223.1250, 239.0625, 255.0000,
     255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000,
     255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000, 255.0000,
     239.0625, 223.1250, 207.1875, 191.2500, 175.3125, 159.3750, 143.4375, 127.5000,
     111.5625, 95.6250,  79.6875,  63.7500,  47.8125,  31.8750,  15.9375,  0,
     0,        0,        0,        0,        0,        0,        0,        0,
     0,        0,        0,        0,        0,        0,        0,        0,
     0,        0,        0,        0,        0,        0,        0,        0}};

void toRGBImage(vpImage<float> in, vpImage<vpRGBa> &out, float min, float max) {
    out.resize(in.getHeight(), in.getWidth());
    float Max = 0.0, Min = 10e10;
    float *pIn;

    if (min == 0 && max == 0) {
        pIn = in.bitmap;
        for (int i = 0; i < in.getHeight() * in.getWidth(); i++, pIn++) {
            if ((*pIn) > Max) {
                Max = (*pIn);
            }
            if ((*pIn) < Min) {
                Min = (*pIn);
            }
        }
    } else {
        Min = min;
        Max = max;
    }

    vpRGBa *pOut;
    pOut = out.bitmap;
    pIn = in.bitmap;
    const float range = Max - Min;
    if (range < 1e-10f) {
        for (int i = 0; i < in.getHeight() * in.getWidth(); i++, pOut++)
            *pOut = vpRGBa(0, 0, 0, 255);
        return;
    }
    for (int i = 0; i < in.getHeight() * in.getWidth(); i++, pOut++, pIn++) {
        if (*pIn <= Min || *pIn >= Max) {
            *pOut = vpRGBa(0, 0, 0, 255);
        } else {
            int idx = (int)((64.0f / range) * (*pIn) - (Min * 64.0f / range));
            if (idx < 0) idx = 0;
            if (idx > 63) idx = 63;
            (*pOut).R = RGB[0][idx];
            (*pOut).G = RGB[1][idx];
            (*pOut).B = RGB[2][idx];
        }
    }
}

/***************************************/
/**************ROS <-> VISP*************/
/***************************************/

vpImage<float> imageMsgToVpImageFloat(const sensor_msgs::Image &Imsg) {
    vpImage<float> I(Imsg.height, Imsg.width);
    for (int v = 0; v < Imsg.height; v++) {
        for (int u = 0; u < Imsg.width; u++) {
            float *dataPtr = (float *)(Imsg.data.data() + (v * Imsg.step + u * sizeof(float)));
            I[v][u] = (*dataPtr);
        }
    }
    return I;
}
