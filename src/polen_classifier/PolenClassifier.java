package polen_classifier;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import org.apache.log4j.Logger;

import surfExtractor.bow_classifier.Bow;
import surfExtractor.bow_classifier.Histogram;
import surfExtractor.clustering.Cluster;
import surfExtractor.clustering.Clustering;
import surfExtractor.exporter.Exporter;
import surfExtractor.image_set.ImageSet;
import surfExtractor.surf_extractor.SurfExtractor;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import configuration.Configuration;

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

		Configuration.setConfiguration("kmeans.kvalue", "768");
		Configuration.setConfiguration("kmeans.iteration", "5");
		Configuration.setConfiguration("imageset.path", "C:\\training");
		Configuration.setConfiguration("arff.path", "c:\\training.arff");
		Configuration.setConfiguration("arff.relation", "training");
		Configuration.setConfiguration("random.seed", "1");

		Configuration.setConfiguration("testimageset.path", "C:\\test");
		Configuration.setConfiguration("testarff.relation", "testing_arff");
		Configuration.setConfiguration("testarff.path", "c:\\testing.arff");
		Configuration.setConfiguration("clusters.path", "c:\\clusters.cluster");
		Configuration.debugParameters();

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
		//clustering.cluster();

		// Export clusters
		LOGGER.info("Saving clusters to file");
		//clustering.saveClustersToFile(new File(Configuration.getConfiguration("clusters.path")));

		// Return final clusters
		//ArrayList<Cluster> featureCluster = clustering.getClusters();

		
		//Load clusters from poln23e 
		ArrayList<Cluster> featureCluster = Clustering.loadClustersFromFile(new File("C:\\Users\\Hugo\\Desktop\\clusters.cluster"));
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
		//bow = new Bow(is, new File(Configuration.getConfiguration("clusters.path")));
		bow = new Bow(is, featureCluster);

		// Compute frequency histograms
		bow.computeHistograms();

		// Return frequency histograms
		h = bow.getHistograms();
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
		 */
		BufferedReader reader = new BufferedReader(new FileReader(Configuration.getConfiguration("arff.path")));
		Instances train = new Instances(reader);
		train.setClassIndex(train.numAttributes() - 1);

		reader = new BufferedReader(new FileReader(Configuration.getConfiguration("testarff.path")));
		Instances test = new Instances(reader);
		test.setClassIndex(test.numAttributes() - 1);

		SMO smo = new SMO();
		try {
			smo.buildClassifier(train);
		} catch (Exception e) {
			e.printStackTrace();
		}

		Evaluation eval = null;
		try {
			eval = new Evaluation(train);
			eval.evaluateModel(smo, test);
		} catch (Exception e) {
			e.printStackTrace();
		}

		LOGGER.info(eval.toSummaryString("\nResults\n======\n", false));
		try {
			LOGGER.info(eval.toMatrixString());
			FastVector fv = eval.predictions();
			for(int i = 0; i < fv.size(); i++) {
				LOGGER.info(fv.elementAt(i));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		

	}
}
