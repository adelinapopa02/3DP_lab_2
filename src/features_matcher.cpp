#include "features_matcher.h"

#include <iostream>
#include <map>
#include <opencv2/dnn.hpp>

FeatureMatcher::FeatureMatcher(cv::Mat intrinsics_matrix, cv::Mat dist_coeffs,
                               bool use_modern_features, double focal_scale) :
  use_modern_features_(use_modern_features)
{
  intrinsics_matrix_ = intrinsics_matrix.clone();
  dist_coeffs_ = dist_coeffs.clone();
  new_intrinsics_matrix_ = intrinsics_matrix.clone();
  new_intrinsics_matrix_.at<double>(0,0) *= focal_scale;
  new_intrinsics_matrix_.at<double>(1,1) *= focal_scale;
}

cv::Mat FeatureMatcher::readUndistortedImage(const std::string& filename )
{
  cv::Mat img = cv::imread(filename), und_img, dbg_img;
  cv::undistort	(	img, und_img, intrinsics_matrix_, dist_coeffs_, new_intrinsics_matrix_ );

  return und_img;
}

void FeatureMatcher::extractFeatures()
{
  features_.resize(images_names_.size());
  descriptors_.resize(images_names_.size());
  feats_colors_.resize(images_names_.size());

  auto orb_detector = cv::ORB::create(10000, 1.2, 8);

  for( int i = 0; i < images_names_.size(); i++  )
  {
    std::cout<<"Computing descriptors for image "<<i<<std::endl;
    cv::Mat img = readUndistortedImage(images_names_[i]);


    //////////////////////////// Code to be completed (2/7) /////////////////////////////////
    // Extract salient points + descriptors from i-th image.
    //
    // A standard implementation (else branch) that uses the ORB features is already provided.
    // It stores them into the features_[i] and descriptors_[i] vectors, and extract the
    // color (cv::Vec3b) of each feature and store in feats_colors_[i] vector.
    //
    // You are required to implement an alternative, more modern feature detection and
    // description scheme inside the if (use_modern_features_) branch (e.g., by means the
    // loadExternalFeatures() function). Examples are SuperPoint
    // (https://github.com/eric-yyjau/pytorch-superpoint), DISK
    // (https://github.com/cvlab-epfl/disk) or ALIKED (https://github.com/Shiaoming/ALIKED),
    // or other alternatives.
    //
    // IMPORTANT: You also need to update the matching part, see the branch
    // if (use_modern_features_) in the FeatureMatcher::exhaustiveMatching() method.
    // For some methods, the feature description and matching phase are merged,
    // so you may only need to change Feature Matcher::exhaustive Matching()

    if (use_modern_features_)
    {
      // OPTION A: Inference inside C++
      // 1. Load a pre-trained model (e.g., SuperPoint.onnx) using cv::dnn::readNet().
      // 2. Convert 'img' to a blob and run net.forward().
      // 3. Post-process the output tensors to fill features_[i] and descriptors_[i].
      // See for example:
      // https://docs.opencv.org/4.x/dd/d55/pytorch_cls_c_tutorial_dnn_conversion.html
      // WARNING: By default, cv::dnn run in CPU only

      // OPTION B: Data Loading (Fallback)
      // If local hardware doesn't support inference, implement loadExternalFeatures()
      // to read keypoints and descriptors from a file (e.g., .txt) generated
      // beforehand by a Python script on your dataset.
      // loadExternalFeatures(image_path, features_[i], descriptors_[i]);

      // Remeber to Look-up features colors!

      static cv::dnn::Net superpoint_net;
      static bool net_loaded = false;
      static bool net_failed = false;

      if (!net_loaded && !net_failed)
      {
        try {
          superpoint_net = cv::dnn::readNet("../superpoint.onnx");
          superpoint_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
          superpoint_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
          net_loaded = true;
          std::cout << "[SuperPoint] Model loaded successfully." << std::endl;
        } catch (const cv::Exception& e) {
          std::cerr << "[SuperPoint] Failed to load model: " << e.what() << "\n  -> Falling back to ORB." << std::endl;
          net_failed = true;
        }
      }

      if (net_loaded)
      {
        // Pre-process
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_32F, 1.0 / 255.0);

        int H = (gray.rows / 8) * 8;
        int W = (gray.cols  / 8) * 8;
        cv::Mat gray_in;
        cv::resize(gray, gray_in, cv::Size(W, H));

        int Hc = H / 8, Wc = W / 8;

        // Build blob and run inference
        cv::Mat blob = cv::dnn::blobFromImage(gray_in);
        superpoint_net.setInput(blob);

        std::vector<std::string> out_names = superpoint_net.getUnconnectedOutLayersNames();
        std::vector<cv::Mat> outs;
        superpoint_net.forward(outs, out_names);

        // Identify score / descriptor tensors by channel count
        cv::Mat scores_raw, desc_raw;
        for (auto& out : outs) {
          if (out.dims >= 4 && out.size[1] == 256)
            desc_raw = out;
          else
            scores_raw = out;
        }

        if (scores_raw.empty() || desc_raw.empty())
        {
          std::cerr << "[SuperPoint] Unexpected output layout, using ORB." << std::endl;
          goto use_orb;
        }

        // Decode score heatmap
        cv::Mat heatmap(H, W, CV_32F, 0.0f);

        auto get4d = [](const cv::Mat& m, int ch, int r, int c) -> float {
          int rows = m.size[2], cols = m.size[3];
          return m.ptr<float>()[ch * rows * cols + r * cols + c];
        };

        if (scores_raw.size[1] == 65)
        {
          for (int r = 0; r < Hc; r++)
            for (int c = 0; c < Wc; c++)
            {
              float mx = -1e9f;
              for (int ch = 0; ch < 65; ch++)
                mx = std::max(mx, get4d(scores_raw, ch, r, c));
              float s = 0.f;
              std::vector<float> ex(65);
              for (int ch = 0; ch < 65; ch++) {
                ex[ch] = std::exp(get4d(scores_raw, ch, r, c) - mx);
                s += ex[ch];
              }
              for (int dy = 0; dy < 8; dy++)
                for (int dx = 0; dx < 8; dx++) {
                  int ch = dy * 8 + dx;
                  int py = r * 8 + dy, px = c * 8 + dx;
                  if (py < H && px < W)
                    heatmap.at<float>(py, px) = ex[ch] / s;
                }
            }
        }
        else
        {
          for (int r = 0; r < H; r++)
            for (int c = 0; c < W; c++)
              heatmap.at<float>(r, c) = get4d(scores_raw, 0, r, c);
        }

        // NMS: keep local maxima above threshold
        const int NMS_R = 4;
        const float SCORE_THR = 0.015f;
        const int MAX_KP = 1000;

        std::vector<std::pair<float, cv::Point>> scored_pts;
        for (int r = NMS_R; r < H - NMS_R; r++)
          for (int c = NMS_R; c < W - NMS_R; c++)
          {
            float v = heatmap.at<float>(r, c);
            if (v < SCORE_THR) continue;
            bool is_max = true;
            for (int dr = -NMS_R; dr <= NMS_R && is_max; dr++)
              for (int dc = -NMS_R; dc <= NMS_R && is_max; dc++)
                if ((dr || dc) && heatmap.at<float>(r+dr, c+dc) >= v)
                  is_max = false;
            if (is_max)
              scored_pts.push_back({v, cv::Point(c, r)});
          }

        std::sort(scored_pts.begin(), scored_pts.end(),
                  [](auto& a, auto& b){ return a.first > b.first; });
        if ((int)scored_pts.size() > MAX_KP)
          scored_pts.resize(MAX_KP);

        float sx = (float)img.cols / W;
        float sy = (float)img.rows / H;

        // Sample + L2-normalise descriptors
        int nkp = (int)scored_pts.size();
        cv::Mat desc_out(nkp, 256, CV_32F);

        for (int k = 0; k < nkp; k++)
        {
          float fpx = scored_pts[k].second.x / 8.0f;
          float fpy = scored_pts[k].second.y / 8.0f;
          fpx = std::min(std::max(fpx, 0.f), (float)(Wc - 1));
          fpy = std::min(std::max(fpy, 0.f), (float)(Hc - 1));

          int x0 = (int)fpx, y0 = (int)fpy;
          int x1 = std::min(x0+1, Wc-1), y1 = std::min(y0+1, Hc-1);
          float wx = fpx - x0, wy = fpy - y0;

          float* row = desc_out.ptr<float>(k);
          auto getd = [&](int d, int r, int c) -> float {
            return desc_raw.ptr<float>()[d * Hc * Wc + r * Wc + c];
          };

          float nsq = 0.f;
          for (int d = 0; d < 256; d++) {
            float v = (1-wy)*(1-wx)*getd(d,y0,x0)
                    + (1-wy)*   wx *getd(d,y0,x1)
                    +    wy *(1-wx)*getd(d,y1,x0)
                    +    wy *   wx *getd(d,y1,x1);
            row[d] = v;  nsq += v*v;
          }
          float inv = (nsq > 1e-10f) ? 1.f / std::sqrt(nsq) : 0.f;
          for (int d = 0; d < 256; d++) row[d] *= inv;

          // Store keypoint in original image space + colour
          float ox = scored_pts[k].second.x * sx;
          float oy = scored_pts[k].second.y * sy;
          features_[i].emplace_back(cv::KeyPoint(ox, oy, 8.f, -1.f, scored_pts[k].first));
        }

        descriptors_[i] = desc_out;

        feats_colors_[i].reserve(features_[i].size());
        for (auto& f : features_[i]) {
          int px = std::min(std::max(cvRound(f.pt.x), 0), img.cols-1);
          int py = std::min(std::max(cvRound(f.pt.y), 0), img.rows-1);
          feats_colors_[i].emplace_back(img.at<cv::Vec3b>(py, px));
        }

        std::cout << "[SuperPoint] " << features_[i].size() << " keypoints for image " << i << std::endl;
      }
      else
      {
        // ORB fallback when SuperPoint model is unavailable
        use_orb:
        orb_detector->detectAndCompute(img, cv::Mat(), features_[i], descriptors_[i]);
        feats_colors_[i].reserve(features_[i].size());
        for (auto& f : features_[i])
          feats_colors_[i].emplace_back(img.at<cv::Vec3b>(
              std::min(std::max(cvRound(f.pt.y),0),img.rows-1),
              std::min(std::max(cvRound(f.pt.x),0),img.cols-1)));
      }
    }
    else
    {
      // Standard ready to use ORB implementation
      orb_detector->detectAndCompute(img, cv::Mat(), features_[i], descriptors_[i]);

      // Look-up features colors
      feats_colors_[i].reserve(features_[i].size());
      for( auto &f : features_[i])
      {
        feats_colors_[i].emplace_back(img.at<cv::Vec3b>(f.pt.y, f.pt.x));
      }
    }
    /////////////////////////////////////////////////////////////////////////////////////////
  }
}

void FeatureMatcher::exhaustiveMatching()
{  
  for( int i = 0; i < images_names_.size() - 1; i++ )
  {
    for( int j = i + 1; j < images_names_.size(); j++ )
    {
      std::cout<<"Matching image "<<i<<" with image "<<j<<std::endl;
      std::vector<cv::DMatch> matches, inlier_matches;

      if( use_modern_features_ )
      {
        // Modern descriptors (SuperPoint, etc.) are usually float matrices.
        // You may use a BruteForce with L2 distance and enable cross-check for better precision.
        //
        // You could also use a modern Matching Network such as SuperGlue/LightGlue
        // (https://github.com/magicleap/supergluepretrainednetwork) subsequent
        // geometric verification (Code to be completed (1/7)) is not required,
        // since these networks perform both matching and geometric verification.
        // In this case, you may follow OPTION A or OPTION A (see above).
        /////////////////////////////////////////////////////////////////////////////////////////

        auto bf = cv::BFMatcher::create(cv::NORM_L2, false);
        std::vector<std::vector<cv::DMatch>> knn_matches;
        bf->knnMatch(descriptors_[i], descriptors_[j], knn_matches, 2);

        const float RATIO = 0.8f;
        for (auto& m : knn_matches)
          if (m.size() == 2 && m[0].distance < RATIO * m[1].distance)
            matches.push_back(m[0]);        
        /////////////////////////////////////////////////////////////////////////////////////////

      }
      else
      {
        auto matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
        matcher->match(descriptors_[i], descriptors_[j], matches);
      }

      //////////////////////////// Code to be completed (1/7) /////////////////////////////////
      // Perform Geometric Verification of matches, possibly discarding the outliers
      // (remember that features have been extracted from undistorted images that now has
      // new_intrinsics_matrix_ as K matrix and no distortions).
      // As geometric models, use both the Essential matrix and the Homograph matrix,
      // both by setting new_intrinsics_matrix_ as K matrix.
      // As threshold in the functions to estimate both models, you may use 1.0 or similar.
      // Store inlier matches into the inlier_matches vector
      // Do not set matches between two images if the amount of inliers matches
      // (i.e., geomatrically verified matches) is small (say <= 5 matches)
      // In case of success, set the matches with the function:
      //
      // setMatches( i, j, inlier_matches);
      //
      // where i,j matched images indices.
      /////////////////////////////////////////////////////////////////////////////////////////
      
      if ((int)matches.size() < 8)
        continue;

      std::vector<cv::Point2d> pts_i, pts_j;
      for (auto& m : matches) {
        pts_i.push_back(features_[i][m.queryIdx].pt);
        pts_j.push_back(features_[j][m.trainIdx].pt);
      }

      cv::Mat mask_E, mask_H;
      cv::Mat E = cv::findEssentialMat(pts_i, pts_j, new_intrinsics_matrix_, cv::RANSAC, 0.999, 1.0, mask_E);
      cv::Mat H_mat = cv::findHomography(pts_i, pts_j, cv::RANSAC, 1.0, mask_H);

      int n_E = mask_E.empty() ? 0 : cv::countNonZero(mask_E);
      int n_H = mask_H.empty() ? 0 : cv::countNonZero(mask_H);

      std::cout << "  inliers E=" << n_E << "  H=" << n_H << std::endl;

      if (n_E > 5)
      {
        for (int k = 0; k < (int)matches.size(); k++)
          if (mask_E.at<unsigned char>(k))
            inlier_matches.push_back(matches[k]);

        setMatches(i, j, inlier_matches);
      }
      /////////////////////////////////////////////////////////////////////////////////////////
    }
  }
  testMatches(0.5);
}

void FeatureMatcher::writeToFile ( const std::string& filename, bool normalize_points ) const
{
  FILE* fptr = fopen(filename.c_str(), "w");

  if (fptr == NULL) {
    std::cerr << "Error: unable to open file " << filename;
    return;
  };

  fprintf(fptr, "%d %d %d\n", num_poses_, num_points_, num_observations_);

  double *tmp_observations;
  cv::Mat dst_pts;
  if(normalize_points)
  {
    cv::Mat src_obs( num_observations_,1, cv::traits::Type<cv::Vec2d>::value,
                     const_cast<double *>(observations_.data()));
    cv::undistortPoints(src_obs, dst_pts, new_intrinsics_matrix_, cv::Mat());
    tmp_observations = reinterpret_cast<double *>(dst_pts.data);
  }
  else
  {
    tmp_observations = const_cast<double *>(observations_.data());
  }

  for (int i = 0; i < num_observations_; ++i)
  {
    fprintf(fptr, "%d %d", pose_index_[i], point_index_[i]);
    for (int j = 0; j < 2; ++j) {
      fprintf(fptr, " %g", tmp_observations[2 * i + j]);
    }
    fprintf(fptr, "\n");
  }

  if( colors_.size() == 3*num_points_ )
  {
    for (int i = 0; i < num_points_; ++i)
      fprintf(fptr, "%d %d %d\n", colors_[i*3], colors_[i*3 + 1], colors_[i*3 + 2]);
  }

  fclose(fptr);
}

void FeatureMatcher::testMatches( double scale )
{
  // For each pose, prepare a map that reports the pairs [point index, observation index]
  std::vector< std::map<int,int> > cam_observation( num_poses_ );
  for( int i_obs = 0; i_obs < num_observations_; i_obs++ )
  {
    int i_cam = pose_index_[i_obs], i_pt = point_index_[i_obs];
    cam_observation[i_cam][i_pt] = i_obs;
  }

  for( int r = 0; r < num_poses_; r++ )
  {
    for (int c = r + 1; c < num_poses_; c++)
    {
      int num_mathces = 0;
      std::vector<cv::DMatch> matches;
      std::vector<cv::KeyPoint> features0, features1;
      for (auto const &co_iter: cam_observation[r])
      {
        if (cam_observation[c].find(co_iter.first) != cam_observation[c].end())
        {
          features0.emplace_back(observations_[2*co_iter.second],observations_[2*co_iter.second + 1], 0.0);
          features1.emplace_back(observations_[2*cam_observation[c][co_iter.first]],observations_[2*cam_observation[c][co_iter.first] + 1], 0.0);
          matches.emplace_back(num_mathces,num_mathces, 0);
          num_mathces++;
        }
      }
      cv::Mat img0 = readUndistortedImage(images_names_[r]),
          img1 = readUndistortedImage(images_names_[c]),
          dbg_img;

      cv::drawMatches(img0, features0, img1, features1, matches, dbg_img);
      cv::resize(dbg_img, dbg_img, cv::Size(), scale, scale);
      cv::imshow("", dbg_img);
      if (cv::waitKey() == 27)
        return;
    }
  }
}

void FeatureMatcher::setMatches( int pos0_id, int pos1_id, const std::vector<cv::DMatch> &matches )
{

  const auto &features0 = features_[pos0_id];
  const auto &features1 = features_[pos1_id];

  auto pos_iter0 = pose_id_map_.find(pos0_id),
      pos_iter1 = pose_id_map_.find(pos1_id);

  // Already included position?
  if( pos_iter0 == pose_id_map_.end() )
  {
    pose_id_map_[pos0_id] = num_poses_;
    pos0_id = num_poses_++;
  }
  else
    pos0_id = pose_id_map_[pos0_id];

  // Already included position?
  if( pos_iter1 == pose_id_map_.end() )
  {
    pose_id_map_[pos1_id] = num_poses_;
    pos1_id = num_poses_++;
  }
  else
    pos1_id = pose_id_map_[pos1_id];

  for( auto &match:matches)
  {

    // Already included observations?
    uint64_t obs_id0 = poseFeatPairID(pos0_id, match.queryIdx ),
        obs_id1 = poseFeatPairID(pos1_id, match.trainIdx );
    auto pt_iter0 = point_id_map_.find(obs_id0),
        pt_iter1 = point_id_map_.find(obs_id1);
    // New point
    if( pt_iter0 == point_id_map_.end() && pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = num_points_++;
      point_id_map_[obs_id0] = point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);

      // Average color between two corresponding features (suboptimal since we shouls also consider
      // the other observations of the same point in the other images)
      cv::Vec3f color = (cv::Vec3f(feats_colors_[pos0_id][match.queryIdx]) +
                        cv::Vec3f(feats_colors_[pos1_id][match.trainIdx]))/2;

      colors_.push_back(cvRound(color[2]));
      colors_.push_back(cvRound(color[1]));
      colors_.push_back(cvRound(color[0]));

      num_observations_++;
      num_observations_++;
    }
      // New observation
    else if( pt_iter0 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id1];
      point_id_map_[obs_id0] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos0_id);
      observations_.push_back(features0[match.queryIdx].pt.x);
      observations_.push_back(features0[match.queryIdx].pt.y);
      num_observations_++;
    }
    else if( pt_iter1 == point_id_map_.end() )
    {
      int pt_idx = point_id_map_[obs_id0];
      point_id_map_[obs_id1] = pt_idx;

      point_index_.push_back(pt_idx);
      pose_index_.push_back(pos1_id);
      observations_.push_back(features1[match.trainIdx].pt.x);
      observations_.push_back(features1[match.trainIdx].pt.y);
      num_observations_++;
    }
//    else if( pt_iter0->second != pt_iter1->second )
//    {
//      std::cerr<<"Shared observations does not share 3D point!"<<std::endl;
//    }
  }
}
void FeatureMatcher::reset()
{
  point_index_.clear();
  pose_index_.clear();
  observations_.clear();
  colors_.clear();

  num_poses_ = num_points_ = num_observations_ = 0;
}
