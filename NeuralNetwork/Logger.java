package NeuralNetwork;

import java.time.LocalDate;

public class Logger<T extends BaseObject> {
    Class<T> clazz;
    private static final boolean LOG_METHODS = false;
    private static final boolean LOG_INFO = true;
    private static final boolean LOG_DEBUG = false;
    private static final boolean LOG_ERROR = true;

    public Logger(Class<T> clazz) {
        this.clazz = clazz;
    }

    public void debug(String message) {
        if (LOG_DEBUG) {
            log(String.format("[DEBUG] %s", message));
        }
    }

    public void info(String message) {
        if (LOG_INFO) {
            log(String.format("[INFO] %s", message));
        }
    }

    private void log(String message) {
        System.out.println(String.format("%s [%s] %s", LocalDate.now().toString(), this.clazz.getSimpleName(), message));
    }
    private void err(String message) {
        System.err.println(String.format("%s [%s] %s", LocalDate.now().toString(), this.clazz.getSimpleName(), message));
    }

    public void logMethod(String name) {
        if (LOG_METHODS) {
            log(String.format("Start of method %s()", name));
        }
    }

    public void error(String message, Throwable e) {
        if(LOG_ERROR){
            err(String.format("[ERROR] %s \n%s", message, getStackTraceAsString(e)));
        }
    }

    private String getStackTraceAsString(Throwable e){
        StringBuilder ret = new StringBuilder();
        StackTraceElement[] stackTrace = e.getStackTrace();
        ret.append("Stack Trace:\n");
        for (int i = 0; i < stackTrace.length; i++) {
            ret.append("\t").append(stackTrace[i].getClassName()).append(".")
                .append(stackTrace[i].getMethodName()).append("() ")
                .append(stackTrace[i].getFileName()).append(" @ Line: ")
                .append(stackTrace[i].getLineNumber()).append("\n");
        }

        return ret.toString();
    }
}
