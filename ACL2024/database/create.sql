CREATE TABLE IF NOT EXISTS User (
        UserId INTEGER PRIMARY KEY,
        Reputation INTEGER,
        CreationDate TEXT,
        DisplayName TEXT,
        LastAccessDate TEXT,
        WebsiteUrl TEXT,
        Location Text,
        AboutMe Text,
        Views INTEGER,
        UpVotes INTEGER,
        DownVotes INTEGER,
        ProfileImageUrl TEXT,
        AccountId INTEGER
);

CREATE TABLE IF NOT EXISTS Post (
        PostId                INTEGER PRIMARY KEY,
        PostTypeId            INTEGER,
        AcceptedAnswerId      INTEGER,
        CreationDate          TEXT,
        Score                 INTEGER,
        ViewCount             INTEGER,
        Body                  TEXT,
        OwnerUserId           INTEGER,
        LastEditorUserId      INTEGER,
        LastEditorDisplayName TEXT,
        LastEditDate          TEXT,
        Title                 TEXT,
        Tags                  TEXT,
        AnswerCount           INTEGER,
        CommentCount          INTEGER,
        FavoriteCount         INTEGER,
        CommunityOwnedDate    TEXT,
        ContentLicense        TEXT,
        ParentId              INTEGER,
        FOREIGN KEY (OwnerUserId) REFERENCES User (UserId),
        FOREIGN KEY (LastEditorUserId) REFERENCES User (UserId)
);

CREATE TABLE IF NOT EXISTS Comment (
        CommentId INTEGER PRIMARY KEY,
        PostId INTEGER,
        Score INTEGER,
        Body TEXT,
        CreationDate TEXT,
        UserId INTEGER,
        ContentLicense TEXT,
        FOREIGN KEY (PostId) REFERENCES Post (PostId),
        FOREIGN KEY (UserId) REFERENCES User (UserId)

);

CREATE TABLE IF NOT EXISTS Badge (
        BadgeId INTEGER PRIMARY KEY,
        UserId INTEGER,
        Name TEXT,
        CreationDate TEXT,
        Class INTEGER,
        TagBased TEXT,
        FOREIGN KEY (UserId) REFERENCES User (UserId)
);


CREATE TABLE IF NOT EXISTS Tag (
        TagId INTEGER PRIMARY KEY,
        TagName TEXT,
        Count INTEGER
);


CREATE TABLE IF NOT EXISTS Vote (
        VoteId INTEGER PRIMARY KEY,
        PostId INTEGER,
        VoteTypeId INTEGER,
        CreationDate TEXT,
        FOREIGN KEY (PostId) REFERENCES Post (PostId)
);


CREATE INDEX owner_index ON Post (OwnerUserId);
CREATE INDEX commenter ON Comment (UserId);
CREATE INDEX parents ON Post (ParentId);
CREATE INDEX post_owner ON Post (OwnerUserId);
CREATE INDEX post_editor ON Post (LastEditorUserId);
CREATE INDEX post_CreationDate ON Post (CreationDate);
CREATE INDEX post_id_index ON Post (PostId);
CREATE INDEX badge_user_id ON Badge (UserId);