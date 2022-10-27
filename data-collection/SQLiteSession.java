import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class SQLiteSession {
    private final Logger logger = LogManager.getLogger(SQLiteSession.class);
    private Connection conn;
    public SQLiteSession(String url) {
        Configurator.setLevel(logger.getName(), Level.INFO);
        try {
            // create a connection to the database
            conn = DriverManager.getConnection(url);
            logger.info("Connection to SQLite has been established.");
        } catch (SQLException e) {
            logger.error(e.getMessage());
        }
    }

    public void endSession() {
        try {
            if (conn != null) {
                conn.close();
            }
        } catch (SQLException ex) {
            logger.error(ex.getMessage());
        }
    }

    public static void main(String[] args) {
        SQLiteSession s = new SQLiteSession(
        "jdbc:sqlite:..\\stackoverflow.db"
        );
        s.endSession();
    }
}