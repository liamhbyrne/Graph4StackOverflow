import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.*;

public class SQLiteSession {
    private final Logger logger = LogManager.getLogger(SQLiteSession.class);
    private Connection conn;

    private PreparedStatement ps;

    private int maxBatchSize;
    private int currentBatchsize = 0;

    public SQLiteSession(String url) {
        Configurator.setLevel(logger.getName(), Level.WARN);
        try {
            // create a connection to the database
            conn = DriverManager.getConnection(url);
            conn.setAutoCommit(false);
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

    public void createTables() throws SQLException, IOException {
        /**
         * Create tables using the `create.sql` file in the directory
         */
        Statement stmt = conn.createStatement();
        String query = Files.readString(Path.of("create.sql"));
        stmt.executeUpdate(query);
        conn.commit();
    }

    public void setPreparedStatement(String baseQuery, int batchSize) throws SQLException {
        this.ps = conn.prepareStatement(baseQuery);
        this.maxBatchSize = batchSize;
    }

    public PreparedStatement getPreparedStatement() {
        return ps;
    }

    public void processBatch() throws SQLException {
        ps.addBatch();
        if (++currentBatchsize > maxBatchSize) {
            ps.executeBatch();
            currentBatchsize = 0;
            logger.info("Executed batch insert.");
        }
    }

    public void commitChanges() throws SQLException {
        conn.commit();
    }

    public static void main(String[] args) throws SQLException, IOException {
        SQLiteSession s = new SQLiteSession(
                "jdbc:sqlite:..\\stackoverflow.db"
        );
        s.createTables();
        s.endSession();
    }
}