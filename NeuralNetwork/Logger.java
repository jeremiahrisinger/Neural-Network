package NeuralNetwork;

import java.io.PrintStream;
import java.time.LocalDate;

/**
 * <h3>Logger Class</h3><br>
 * This class is used to log 4 diffent types of messages:
 * <ul>
 * <li>Method Calls
 * <li>Debug messages
 * <li>Information messages
 * <li>Error messages
 * </ul>
 * <b>Description:</b><br>
 * <p>
 * Each have their own constant to set whether they should be logged or not.
 * Each logging type has its own method and will print a fomatted log to the
 * terminal in [Date] [[ClassName]] [LoggerLevel] 'message'. Error messages will
 * have the stack trace appended to the end.
 * </p>
 * <b>Warning:</b><br>
 * <p>
 * This class can only be implemented by a Class<T> where T extends BaseObject
 * </p>
 * 
 */
public class Logger<T extends BaseObject> {

    /**
     * Constant for whether the Logger should log method calls.
     */
    private static final boolean LOG_METHODS = false;

    /**
     * Constant for whether the Logger should log information messages.
     */
    private static final boolean LOG_INFO = true;

    /**
     * Constant for whether the Logger should log debugging messages.
     */
    private static final boolean LOG_DEBUG = false;

    /**
     * Constant for whether the Logger should log error messages. Probably should
     * stay true...
     */
    private static final boolean LOG_ERROR = true;

    private Class<T> clazz;

    /**
     * Basic constructor for the Logger class
     * 
     * @param clazz
     */
    public Logger(Class<T> clazz) {
        this.clazz = clazz;
    }

    /**
     * This method is for logging a debug message. The message will only be logged
     * if {@link #LOG_DEBUG} = true
     * 
     * @see #log(String)
     * @param message
     */
    public void debug(String message) {
        if (LOG_DEBUG) {
            log(String.format("[DEBUG] %s", message));
        }
    }

    /**
     * This method is for logging an info message. This message will only be logged
     * if {@link #LOG_INFO} = true
     * 
     * @see #log(String)
     * @param message
     */
    public void info(String message) {
        if (LOG_INFO) {
            log(String.format("[INFO] %s", message));
        }
    }

    /**
     * This is a generic helper logger method. It will log a message to System.out
     * with proper the proper logger format.
     * 
     * @param message
     */
    private void log(String message) {
        formattedLog(message, System.out);
    }

    /**
     * This is a generic helper logger method. It will log a message to System.err
     * with proper the proper logger format.
     * 
     * @param message
     */
    private void err(String message) {
        formattedLog(message, System.err);
    }

    /**
     * Helper method for simplifying other methods and logger formatting
     * 
     * @param message
     * @param printer
     */
    private void formattedLog(String message, PrintStream printer) {
        printer.println(String.format("%s [%s] %s", LocalDate.now().toString(), this.clazz.getSimpleName(), message));
    }

    /**
     * This method is for logging the start of each method. This will only be logged
     * if {@link #LOG_METHODS} = true
     * 
     * @see #log(String)
     * @param name
     */
    public void logMethod(String name) {
        if (LOG_METHODS) {
            log(String.format("Start of method %s()", name));
        }
    }

    /**
     * This method is for logging an error message. This will only be logged if
     * {@link #LOG_ERROR} = true
     * 
     * @see #err(String)
     * @param message
     * @param e
     */
    public void error(String message, Throwable e) {
        if (LOG_ERROR) {
            err(String.format("[ERROR] %s \n%s", message, getStackTraceAsString(e)));
        }
    }

    /**
     * Helper method for logging the error. This method will build a stack trace
     * string for the error logger to append and print.
     * 
     * @param e
     * @return
     */
    private String getStackTraceAsString(Throwable e) {
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
