package polen_classifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import org.apache.log4j.Logger;

import weka.*;
import surfExtractor.*;
import surfExtractor.bow_classifier.Bow;
import surfExtractor.bow_classifier.Histogram;
import surfExtractor.clustering.Cluster;
import surfExtractor.clustering.Clustering;
import surfExtractor.exporter.Exporter;
import surfExtractor.image_set.ImageSet;
import surfExtractor.misc.Configuration;
import surfExtractor.surf_extractor.Main;
import surfExtractor.surf_extractor.SurfExtractor;

public class PolenClassifier {
	private final static Logger LOGGER = Logger.getLogger(PolenClassifier.class);

	public static void main(String[] args) {
		new PolenClassifier();
	}

	public PolenClassifier() {
		try {
			run();
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void run() throws IOException {

		Configuration.addConfiguration("kmeans.kvalue", "768");
		Configuration.addConfiguration("kmeans.iteration", "5");
		Configuration.addConfiguration("imageset.path", "C:\\training");
		Configuration.addConfiguration("arff.path", "c:\\training.arff");
		Configuration.addConfiguration("arff.relation", "training");
		Configuration.addConfiguration("random.seed", "1");
		
		Configuration.addConfiguration("testimageset.path", "C:\\test");
		Configuration.addConfiguration("testarff.relation", "testing_arff");
		Configuration.addConfiguration("testarff.path", "c:\\testing.arff");
		Configuration.addConfiguration("clusters.path", "c:\\clusters.cluster");

		// Load images from ImageSet
		ImageSet is = new ImageSet(Configuration.getConfiguration("imageset.path"));

		// Create clustering object
		Clustering clustering = new Clustering(is, Integer.valueOf(Configuration.getConfiguration("kmeans.kvalue")), Integer.valueOf(Configuration.getConfiguration("kmeans.iteration")));

		// Set Dataset 'name'
		is.setRelation(Configuration.getConfiguration("arff.relation"));

		// Create SURF Feature extractor objects
		SurfExtractor surfExtractor = new SurfExtractor();

		// Load images from ImageSet
		is.getImageClasses();

		// Use surfExtractor to extract SURF features
		surfExtractor.extractImageSet(is);

		// Debug feature number for each image
		/*
		 * for (ImageClass ic : is.getImageClasses()) { for (Image i :
		 * ic.getImages()) { LOGGER.info(i.getFeatures().size() +
		 * " SURF features detected for: " + i.getFile().getName()); } }
		 */

		// Cluster all features
		clustering.cluster();

		// Export clusters
		LOGGER.info("Saving clusters to file");
		clustering.saveClustersToFile(new File(Configuration.getConfiguration("clusters.path")));

		// Return final clusters
		ArrayList<Cluster> featureCluster = clustering.getClusters();

		// Load Bag Of Words classifier
		Bow bow = new Bow(is, featureCluster);

		// Compute frequency histograms
		bow.computeHistograms();

		// Return frequency histograms
		ArrayList<Histogram> h = bow.getHistograms();

		// Debug histograms
		/*
		 * LOGGER.info("Debugging image histograms"); for (Histogram hh : h) {
		 * LOGGER.info("Histogram: " + histogramToString(hh)); }
		 */

		// Write experimental arff
		Exporter exporter = new Exporter(is, bow);
		exporter.generateArffFile(Configuration.getConfiguration("arff.path"));
		
		/*
		 * 
		 * TEST ARFF CODE BELOW
		 * 
		 */
		
		

		// Load images from ImageSet
		is = new ImageSet(Configuration.getConfiguration("testimageset.path"));

		// Set Dataset 'name'
		is.setRelation(Configuration.getConfiguration("testarff.relation"));

		// Load images from ImageSet
		is.getImageClasses();

		// Use surfExtractor to extract SURF features
		surfExtractor.extractImageSet(is);

		// Debug feature number for each image
		/*
		 * for (ImageClass ic : is.getImageClasses()) { for (Image i :
		 * ic.getImages()) { LOGGER.info(i.getFeatures().size() +
		 * " SURF features detected for: " + i.getFile().getName()); } }
		 */

		// Export clusters
		LOGGER.info("Saving clusters to file");

		// Load Bag Of Words classifier
		bow = new Bow(is, new File(Configuration.getConfiguration("clusters.path")));

		// Compute frequency histograms
		bow.computeHistograms();

		// Return frequency histograms
		h = bow.getHistograms();

		// Debug histograms
		/*
		 * LOGGER.info("Debugging image histograms"); for (Histogram hh : h) {
		 * LOGGER.info("Histogram: " + histogramToString(hh)); }
		 */

		// Write experimental arff
		exporter = new Exporter(is, bow);
		exporter.generateArffFile(Configuration.getConfiguration("testarff.path"));
		
		/*
		 * 
		 * CLASSIFICATION PART 
		 * 
		 */
		
		

	}
}
