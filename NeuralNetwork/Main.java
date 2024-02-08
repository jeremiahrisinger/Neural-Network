package NeuralNetwork;

import java.util.ArrayList;
import java.util.List;

public class Main extends BaseObject {

    private static final Logger<Main> LOGGER = new Logger<Main>(Main.class);

    public static void main(String[] args) {
        LOGGER.logMethod("main");
        double[][] X = {
                { 0, 0 },
                { 1, 0 },
                { 0, 1 },
                { 1, 1 }
        };
        double[][] Y = {
                { 0 }, { 1 }, { 1 }, { 0 }
        };

        NeuralNerwork nn = new NeuralNerwork(2, 10, 1);
        nn.fit(X, Y, 5000);

        double[] input = { 0, 1 };
        List<Double> output = new ArrayList<>();
        try {
            output = nn.predict(input);
        } catch (Exception e) {
            LOGGER.error("Error occured while trying to predict output.", e);
        }

        LOGGER.info(input.toString() + " " + output.toString());
    }

}
