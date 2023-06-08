import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.logging.log4j.core.config.Configurator;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import javax.xml.XMLConstants;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.IOException;
import java.sql.SQLException;

public class StackOverflowHandler extends DefaultHandler {
    private final SQLiteSession session;
    private final String baseInsertQuery;

    private final String[] attributeNames;

    private final Logger logger = LogManager.getLogger(StackOverflowHandler.class);

    public StackOverflowHandler(String pathToDB, String baseInsertQuery, String[] attributeNames) {
        Configurator.setLevel(logger.getName(), Level.INFO);
        this.session = new SQLiteSession(pathToDB);
        this.baseInsertQuery = baseInsertQuery;
        this.attributeNames = attributeNames;
    }

    public Logger getLogger() {
        return logger;
    }

    public SQLiteSession getSession() {
        return session;
    }

    @Override
    public void startDocument() {
        try {
            session.setPreparedStatement(baseInsertQuery, 500);
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public void startElement(String uri, String lName, String qName, Attributes attr) {
        if (qName.equals("row")) {
            try {
                for (int i = 0; i < attributeNames.length; i++) {
                    getSession().getPreparedStatement().setString(i+1, attr.getValue(attributeNames[i]));
                }
                getSession().processBatch();

            } catch (SQLException e) {
                getLogger().error(String.format("SQL Error! %s", e));
            }

        } else {
            getLogger().error(String.format("Non-row encountered: %s", qName));
        }
    }

    @Override
    public void endDocument() {
        try {
            session.getPreparedStatement().executeBatch();
            session.commitChanges();
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }
}



class Main {
    public static void main(String[] args) throws ParserConfigurationException, SAXException, IOException {
        final long start = System.currentTimeMillis();
        String PATH_TO_DB = "jdbc:sqlite:..\\stackoverflow.db";

        // New SAXParser
        SAXParserFactory factory = SAXParserFactory.newInstance();
        factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING,false);
        SAXParser saxParser = factory.newSAXParser();

        /*// Tags
        String[] tagAttributeNames = {"Id", "TagName", "Count"};
        StackOverflowHandler tagsHandler = new StackOverflowHandler(PATH_TO_DB, "INSERT INTO Tag VALUES (?, ?, ?)", tagAttributeNames);
        saxParser.parse("..\\data\\raw\\stackoverflow.com-Tags\\tags.xml", tagsHandler);

        // Users
        String[] userAttributeNames = {"Id", "Reputation", "CreationDate", "DisplayName", "LastAccessDate", "WebsiteUrl", "Location", "AboutMe", "Views", "UpVotes", "DownVotes", "ProfileImageUrl", "AccountId"};
        StackOverflowHandler usersHandler = new StackOverflowHandler(PATH_TO_DB, "INSERT INTO User VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", userAttributeNames);
        saxParser.parse("..\\data\\raw\\stackoverflow.com-Users\\users.xml", usersHandler);
*/
        // Posts
        String[] postAttributeNames = {"Id", "PostTypeId", "AcceptedAnswerId", "CreationDate", "Score", "ViewCount", "Body", "OwnerUserId", "LastEditorUserId", "LastEditorDisplayName", "LastEditDate", "Title", "Tags", "AnswerCount", "CommentCount", "FavoriteCount", "CommunityOwnedDate", "ContentLicense", "ParentId"};
        StackOverflowHandler postsHandler = new StackOverflowHandler(PATH_TO_DB, "INSERT INTO Post VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", postAttributeNames);
        saxParser.parse("..\\data\\raw\\stackoverflow.com-Posts\\posts.xml", postsHandler);

        // Comments
        String[] commentAttributeNames = {"Id", "PostId", "Score", "Text", "CreationDate", "UserId", "ContentLicense"};
        StackOverflowHandler commentsHandler = new StackOverflowHandler(PATH_TO_DB, "INSERT INTO Comment VALUES (?, ?, ?, ?, ?, ?, ?)", commentAttributeNames);
        saxParser.parse("..\\data\\raw\\stackoverflow.com-Comments\\comments.xml", commentsHandler);

        // Badges
        String[] badgeAttributeNames = {"Id", "UserId", "Name", "Date", "Class", "TagBased"};
        StackOverflowHandler badgesHandler = new StackOverflowHandler(PATH_TO_DB, "INSERT INTO Badge VALUES (?, ?, ?, ?, ?, ?)", badgeAttributeNames);
        saxParser.parse("..\\data\\raw\\stackoverflow.com-Badges\\badges.xml", badgesHandler);

        // Votes
        String[] voteAttributeNames = {"Id", "PostId", "VoteTypeId", "CreationDate"};
        StackOverflowHandler votesHandler = new StackOverflowHandler(PATH_TO_DB, "INSERT INTO Vote VALUES (?, ?, ?, ?)", voteAttributeNames);
        saxParser.parse("..\\data\\raw\\stackoverflow.com-Votes\\votes.xml", votesHandler);


        // time end
        final long durationInMilliseconds = System.currentTimeMillis()-start;
        System.out.println("SAX parse took " + durationInMilliseconds + "ms.");
    }
}