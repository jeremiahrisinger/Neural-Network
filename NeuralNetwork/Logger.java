package NeuralNetwork;

import java.time.LocalDate;

public class Logger<T extends BaseObject> {
    Class<T> clazz;

    public Logger(Class<T> clazz) {
        this.clazz = clazz;
    }

    public void log(String message) {
        System.out.println(String.format("%s [%s] %s", LocalDate.now().toString(), this.clazz.getName(), message));
    }

    public void logMethod(String name) {
        log(String.format("Start of method %s()", name));
    }

    public void error(String message, Throwable e) {
        System.err.println(String.format("%s [%s] ERROR %s %s", LocalDate.now().toString(), this.clazz.getName(),
                message, e.getMessage()));
        e.printStackTrace(System.err);
    }
}
